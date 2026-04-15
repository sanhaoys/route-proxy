"""上游 HTTP 代理通用逻辑。"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Awaitable

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("route_proxy")

STREAM_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}


def upstream_headers(request: Request) -> dict[str, str]:
    """从客户端请求中提取需要透传给上游的 headers。"""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if auth := request.headers.get("authorization"):
        headers["Authorization"] = auth
    return headers


# ─────────────────── 流式：先检查状态码再决定返回类型 ───────────────────


async def stream_and_forward(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    headers: dict,
    converter: Callable[..., AsyncIterator[str]],
) -> StreamingResponse | JSONResponse:
    """先检查上游状态码，错误返回 HTTP 错误，成功则流式转发。"""
    log.info("stream_forward model=%s url=%s", body.get("model"), url)
    resp = await client.send(
        client.build_request("POST", url, json=body, headers=headers),
        stream=True,
    )
    if resp.status_code != 200:
        error = await _read_error(resp)
        await resp.aclose()
        log.warning("upstream error status=%d model=%s", resp.status_code, body.get("model"))
        return JSONResponse(status_code=resp.status_code, content=error)

    async def generate():
        try:
            async for event in converter(resp):
                yield event
        except Exception:
            log.exception("stream converter error model=%s", body.get("model"))
            raise
        finally:
            await resp.aclose()

    return StreamingResponse(generate(), media_type="text/event-stream", headers=STREAM_HEADERS)


async def passthrough_stream(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    headers: dict,
) -> StreamingResponse | JSONResponse:
    """先检查上游状态码，成功则直接透传 SSE 流。"""
    log.info("passthrough model=%s url=%s", body.get("model"), url)
    resp = await client.send(
        client.build_request("POST", url, json=body, headers=headers),
        stream=True,
    )
    if resp.status_code != 200:
        error = await _read_error(resp)
        await resp.aclose()
        log.warning("upstream error status=%d model=%s", resp.status_code, body.get("model"))
        return JSONResponse(status_code=resp.status_code, content=error)

    async def generate():
        try:
            async for line in resp.aiter_lines():
                if line.strip():
                    yield line + "\n\n"
        finally:
            await resp.aclose()

    return StreamingResponse(generate(), media_type="text/event-stream", headers=STREAM_HEADERS)


# ─────────────────── 非流式：收集 ───────────────────


async def stream_and_collect(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    headers: dict,
    collector: Callable[..., Awaitable[dict]],
) -> JSONResponse:
    """打开上游流，用 collector 收集全部 chunk 后返回 JSON。"""
    log.info("collect model=%s url=%s", body.get("model"), url)
    async with client.stream("POST", url, json=body, headers=headers) as resp:
        if resp.status_code != 200:
            log.warning("upstream error status=%d model=%s", resp.status_code, body.get("model"))
            return JSONResponse(
                status_code=resp.status_code,
                content=await _read_error(resp),
            )
        return JSONResponse(content=await collector(resp))


# ─────────────────── 内部 ───────────────────


async def _read_error(resp: httpx.Response) -> dict:
    raw = await resp.aread()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": {"message": raw.decode(errors="replace")}}
