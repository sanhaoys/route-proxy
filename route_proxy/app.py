"""FastAPI 应用与路由定义。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("route_proxy")
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import BASE_URL, CC_URL, TIMEOUT
from .convert_request import convert_request
from .convert_response import (
    collect_and_build_response,
    collect_chat_completion,
    stream_response_events,
)
from .proxy import (
    passthrough_stream,
    stream_and_collect,
    stream_and_forward,
    upstream_headers,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=TIMEOUT)
    yield
    await app.state.client.aclose()


app = FastAPI(title="Route Proxy", lifespan=lifespan)


# ───────────────────── /v1/responses ─────────────────────


@app.post("/v1/responses")
async def handle_responses(request: Request):
    body = await request.json()
    want_stream = body.get("stream", False)

    cc_body = convert_request(body)
    cc_body["stream"] = True
    cc_body["stream_options"] = {"include_usage": True}

    headers = upstream_headers(request)
    client: httpx.AsyncClient = request.app.state.client

    log.info("POST /v1/responses model=%s stream=%s", body.get("model"), want_stream)

    if want_stream:
        return await stream_and_forward(
            client, CC_URL, cc_body, headers,
            converter=lambda resp: stream_response_events(resp, body),
        )

    return await stream_and_collect(
        client, CC_URL, cc_body, headers,
        collector=lambda resp: collect_and_build_response(resp, body),
    )


# ───────────────────── /v1/chat/completions ─────────────────────


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    body = await request.json()
    want_stream = body.get("stream", False)

    log.info("POST /v1/chat/completions model=%s stream=%s", body.get("model"), want_stream)

    body["stream"] = True
    body.setdefault("stream_options", {})["include_usage"] = True

    headers = upstream_headers(request)
    client: httpx.AsyncClient = request.app.state.client

    if want_stream:
        return await passthrough_stream(client, CC_URL, body, headers)

    return await stream_and_collect(
        client, CC_URL, body, headers,
        collector=collect_chat_completion,
    )


# ───────────────────── 通用透传 ─────────────────────


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_passthrough(request: Request, path: str):
    client: httpx.AsyncClient = request.app.state.client
    headers = upstream_headers(request)

    url = f"{BASE_URL}/v1/{path}"
    kwargs: dict = {"headers": headers, "params": dict(request.query_params)}
    if request.method in ("POST", "PUT"):
        kwargs["content"] = await request.body()

    resp = await client.request(request.method, url, **kwargs)

    content_type = resp.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    return JSONResponse(status_code=resp.status_code, content={"raw": resp.text})
