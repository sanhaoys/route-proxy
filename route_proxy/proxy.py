"""上游 HTTP 代理通用逻辑。

将「打开上游流 → 错误处理 → 转换/收集」这一重复模式抽取到此处。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Awaitable

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

STREAM_RESPONSE_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}


def upstream_headers(request: Request) -> dict[str, str]:
    """从客户端请求中提取需要透传给上游的 headers。"""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if auth := request.headers.get("authorization"):
        headers["Authorization"] = auth
    return headers


# ─────────────────── 核心代理模式 ───────────────────


async def stream_and_forward(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    headers: dict,
    converter: Callable[..., AsyncIterator[str]],
) -> AsyncIterator[str]:
    """打开上游流，用 converter 逐事件转换后 yield 给客户端。

    converter 签名: async def converter(response) -> AsyncIterator[str]
    """
    async with client.stream("POST", url, json=body, headers=headers) as resp:
        if resp.status_code != 200:
            yield _format_error_sse(await _read_error(resp))
            return
        async for event in converter(resp):
            yield event


async def stream_and_collect(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    headers: dict,
    collector: Callable[..., Awaitable[dict]],
) -> JSONResponse:
    """打开上游流，用 collector 收集全部 chunk 后返回 JSON。

    collector 签名: async def collector(response) -> dict
    """
    async with client.stream("POST", url, json=body, headers=headers) as resp:
        if resp.status_code != 200:
            return JSONResponse(
                status_code=resp.status_code,
                content=await _read_error(resp),
            )
        return JSONResponse(content=await collector(resp))


def streaming_response(gen: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers=STREAM_RESPONSE_HEADERS,
    )


# ─────────────────── 透传 ───────────────────


async def passthrough_sse(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    headers: dict,
) -> AsyncIterator[str]:
    """直接透传上游 SSE 流，不做格式转换。"""
    async with client.stream("POST", url, json=body, headers=headers) as resp:
        async for line in resp.aiter_lines():
            if line.strip():
                yield line + "\n\n"


# ─────────────────── 内部 ───────────────────


async def _read_error(resp: httpx.Response) -> dict:
    raw = await resp.aread()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": {"message": raw.decode(errors="replace")}}


def _format_error_sse(error: dict) -> str:
    return f"event: error\ndata: {json.dumps(error, ensure_ascii=False)}\n\n"
