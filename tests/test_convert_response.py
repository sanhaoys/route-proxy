"""测试 Chat Completions -> Responses API 响应转换。"""

import asyncio
import json

from route_proxy.convert_response import (
    collect_and_build_response,
    collect_chat_completion,
    stream_response_events,
)


class FakeResponse:
    """模拟 httpx 流式响应的 aiter_lines。"""

    def __init__(self, chunks: list[dict], include_done: bool = True):
        self._lines: list[str] = []
        for c in chunks:
            self._lines.append(f"data: {json.dumps(c)}")
            self._lines.append("")
        if include_done:
            self._lines.append("data: [DONE]")
            self._lines.append("")

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _parse_sse_events(raw_events: list[str]) -> list[tuple[str, dict]]:
    """解析 data-only SSE 事件列表，返回 (type, data) 元组。"""
    parsed = []
    for raw in raw_events:
        for part in raw.strip().split("\n\n"):
            part = part.strip()
            if not part:
                continue
            # data-only 格式：只有 data: 行
            for line in part.split("\n"):
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    parsed.append((data.get("type"), data))
    return parsed


# ─────── 非流式 Responses ───────


def test_collect_text_response():
    chunks = [
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    ]
    resp = FakeResponse(chunks)
    result = _run(collect_and_build_response(resp, {"model": "gpt-4o"}))

    assert result["object"] == "response"
    assert result["status"] == "completed"
    assert result["output_text"] == "Hello world"
    assert result["output"][0]["type"] == "message"
    assert result["output"][0]["content"][0]["text"] == "Hello world"
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 5


def test_collect_tool_call_response():
    chunks = [
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}, "finish_reason": None}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]}, "finish_reason": None}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '":"Tokyo"}'}}]}, "finish_reason": None}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}},
    ]
    resp = FakeResponse(chunks)
    result = _run(collect_and_build_response(resp, {"model": "gpt-4o"}))

    assert result["status"] == "completed"
    assert len(result["output"]) == 1
    fc = result["output"][0]
    assert fc["type"] == "function_call"
    assert fc["name"] == "get_weather"
    assert json.loads(fc["arguments"]) == {"loc": "Tokyo"}


def test_collect_length_finish():
    chunks = [
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
    ]
    resp = FakeResponse(chunks)
    result = _run(collect_and_build_response(resp, {"model": "gpt-4o"}))

    assert result["status"] == "incomplete"
    assert result["incomplete_details"]["reason"] == "max_output_tokens"


# ─────── 流式 Responses ───────


def test_stream_data_only_format():
    """验证 SSE 事件是 data-only 格式，包含 type 和 sequence_number。"""
    chunks = [
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}},
    ]
    resp = FakeResponse(chunks)

    raw_events = []

    async def collect():
        async for e in stream_response_events(resp, {"model": "gpt-4o"}):
            raw_events.append(e)

    _run(collect())

    # 验证同时有 event: 行和 data: 行
    for raw in raw_events:
        assert "event: " in raw
        assert "data: " in raw

    parsed = _parse_sse_events(raw_events)

    # 所有事件都有 type 和 sequence_number
    for event_type, data in parsed:
        assert event_type is not None
        assert "sequence_number" in data

    event_types = [e[0] for e in parsed]

    assert "response.created" in event_types
    assert "response.in_progress" in event_types
    assert "response.output_item.added" in event_types
    assert "response.content_part.added" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.output_text.done" in event_types
    assert "response.content_part.done" in event_types
    assert "response.output_item.done" in event_types
    assert "response.completed" in event_types


def test_stream_lifecycle_events_nest_response():
    """验证生命周期事件把 response 对象嵌套在 response 键下。"""
    chunks = [
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "cc-1", "model": "gpt-4o", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}},
    ]
    resp = FakeResponse(chunks)

    raw_events = []

    async def collect():
        async for e in stream_response_events(resp, {"model": "gpt-4o"}):
            raw_events.append(e)

    _run(collect())
    parsed = _parse_sse_events(raw_events)

    # response.created 应包含 response 键
    created = [d for t, d in parsed if t == "response.created"][0]
    assert "response" in created
    assert created["response"]["object"] == "response"
    assert created["response"]["status"] == "queued"

    # response.completed 应包含 response 键
    completed = [d for t, d in parsed if t == "response.completed"][0]
    assert "response" in completed
    assert completed["response"]["status"] == "completed"
    assert completed["response"]["usage"]["input_tokens"] == 5

    # delta 事件不嵌套 response
    delta = [d for t, d in parsed if t == "response.output_text.delta"][0]
    assert "response" not in delta
    assert delta["delta"] == "Hi"


# ─────── CC 非流式收集 ───────


def test_collect_chat_completion():
    chunks = [
        {"id": "chatcmpl-1", "model": "gpt-4o", "system_fingerprint": "fp_abc", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {"id": "chatcmpl-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": None}]},
        {"id": "chatcmpl-1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "chatcmpl-1", "model": "gpt-4o", "choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9}},
    ]
    resp = FakeResponse(chunks)
    result = _run(collect_chat_completion(resp))

    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "OK"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 8
    assert result["system_fingerprint"] == "fp_abc"
