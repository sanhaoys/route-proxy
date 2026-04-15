"""Chat Completions response -> Responses API response.

支持两种模式：
1. collect  — 收集所有 stream chunk，拼装成完整的 Responses 对象
2. stream   — 将 CC chunk 实时转换为 Responses 语义化 SSE 事件
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator


# ─────────────────────── helpers ───────────────────────


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def parse_sse_stream(response) -> AsyncIterator[dict]:
    """解析上游 Chat Completions 的 SSE data 行。"""
    async for line in response.aiter_lines():
        line = line.strip()
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


# ─────────────────────── 非流式：收集并构建 ───────────────────────


async def collect_and_build_response(response, original_body: dict) -> dict:
    """收集所有 CC stream chunk，拼装成完整的 Responses API 响应。"""
    resp_id = _gen_id("resp")
    msg_id = _gen_id("msg")

    full_content = ""
    tool_calls: dict[int, dict] = {}
    model = original_body.get("model", "")
    usage: dict = {}
    finish_reason = "stop"

    async for chunk in parse_sse_stream(response):
        model = chunk.get("model", model)

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})

            if content := delta.get("content"):
                full_content += content

            if tc_list := delta.get("tool_calls"):
                for tc in tc_list:
                    idx = tc["index"]
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.get("id"):
                        tool_calls[idx]["id"] = tc["id"]
                    if name := tc.get("function", {}).get("name"):
                        tool_calls[idx]["name"] = name
                    if args := tc.get("function", {}).get("arguments"):
                        tool_calls[idx]["arguments"] += args

            if fr := choice.get("finish_reason"):
                finish_reason = fr

        if u := chunk.get("usage"):
            usage = u

    # ── 构建 output ──
    output: list[dict] = []

    if full_content:
        output.append(
            {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": full_content, "annotations": []}],
            }
        )

    for idx in sorted(tool_calls):
        tc = tool_calls[idx]
        output.append(
            {
                "id": _gen_id("fc"),
                "type": "function_call",
                "call_id": tc["id"],
                "name": tc["name"],
                "arguments": tc["arguments"],
                "status": "completed",
            }
        )

    # ── status 映射 ──
    status = "completed"
    incomplete_details = None
    if finish_reason == "length":
        status = "incomplete"
        incomplete_details = {"reason": "max_output_tokens"}
    elif finish_reason == "content_filter":
        status = "failed"

    return {
        "id": resp_id,
        "object": "response",
        "created_at": time.time(),
        "model": model,
        "status": status,
        "error": None,
        "incomplete_details": incomplete_details,
        "instructions": original_body.get("instructions"),
        "metadata": original_body.get("metadata", {}),
        "output": output,
        "output_text": full_content,
        "parallel_tool_calls": original_body.get("parallel_tool_calls"),
        "temperature": original_body.get("temperature", 1.0),
        "tool_choice": original_body.get("tool_choice"),
        "tools": original_body.get("tools", []),
        "top_p": original_body.get("top_p", 1.0),
        "max_output_tokens": original_body.get("max_output_tokens"),
        "previous_response_id": original_body.get("previous_response_id"),
        "reasoning": original_body.get("reasoning"),
        "text": original_body.get("text"),
        "truncation": original_body.get("truncation"),
        "usage": _convert_usage(usage),
        "user": original_body.get("user"),
    }


# ─────────────────────── 流式：事件转换 ───────────────────────


async def stream_response_events(response, original_body: dict) -> AsyncIterator[str]:
    """将 CC stream chunk 实时转换为 Responses API 的语义化 SSE 事件流。"""
    resp_id = _gen_id("resp")
    msg_id = _gen_id("msg")
    now = time.time()
    model = original_body.get("model", "")

    # 状态追踪
    message_added = False
    content_started = False
    finished = False
    completed = False
    full_content = ""
    tool_calls: dict[int, dict] = {}  # idx -> {id, name, arguments}
    tool_call_item_ids: dict[int, str] = {}  # idx -> item_id

    # ── response.created ──
    base = {
        "id": resp_id,
        "object": "response",
        "created_at": now,
        "model": model,
        "status": "queued",
        "output": [],
    }
    yield _sse("response.created", base)

    base["status"] = "in_progress"
    yield _sse("response.in_progress", base)

    async for chunk in parse_sse_stream(response):
        model = chunk.get("model", model)

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # ── 文本内容 ──
            if (content := delta.get("content")) is not None:
                if not message_added:
                    message_added = True
                    yield _sse(
                        "response.output_item.added",
                        {
                            "output_index": 0,
                            "item": {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                        },
                    )

                if not content_started:
                    content_started = True
                    yield _sse(
                        "response.content_part.added",
                        {
                            "item_id": msg_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        },
                    )

                if content:
                    full_content += content
                    yield _sse(
                        "response.output_text.delta",
                        {
                            "item_id": msg_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": content,
                        },
                    )

            # ── 工具调用 ──
            if tc_list := delta.get("tool_calls"):
                for tc in tc_list:
                    idx = tc["index"]
                    output_idx = (1 if message_added else 0) + idx

                    if idx not in tool_calls:
                        item_id = _gen_id("fc")
                        tool_calls[idx] = {
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": "",
                        }
                        tool_call_item_ids[idx] = item_id

                        yield _sse(
                            "response.output_item.added",
                            {
                                "output_index": output_idx,
                                "item": {
                                    "id": item_id,
                                    "type": "function_call",
                                    "call_id": tc.get("id", ""),
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": "",
                                    "status": "in_progress",
                                },
                            },
                        )

                    if tc.get("id"):
                        tool_calls[idx]["id"] = tc["id"]
                    if name := tc.get("function", {}).get("name"):
                        tool_calls[idx]["name"] = name

                    if args := tc.get("function", {}).get("arguments"):
                        tool_calls[idx]["arguments"] += args
                        yield _sse(
                            "response.function_call_arguments.delta",
                            {
                                "item_id": tool_call_item_ids[idx],
                                "output_index": output_idx,
                                "delta": args,
                            },
                        )

            # ── finish ──
            if finish_reason and not finished:
                finished = True
                for ev in _emit_close_events(
                    msg_id,
                    message_added,
                    content_started,
                    full_content,
                    tool_calls,
                    tool_call_item_ids,
                ):
                    yield ev

        # ── usage（最后一个 chunk） ──
        if u := chunk.get("usage"):
            completed = True
            output_items = _build_final_output(
                msg_id, message_added, content_started, full_content, tool_calls, tool_call_item_ids
            )
            yield _sse(
                "response.completed",
                {
                    "id": resp_id,
                    "object": "response",
                    "created_at": now,
                    "model": model,
                    "status": "completed",
                    "output": output_items,
                    "usage": _convert_usage(u),
                },
            )

    # ── 兜底：上游未发 finish/usage 时补齐事件 ──
    if not finished and (message_added or tool_calls):
        for ev in _emit_close_events(
            msg_id, message_added, content_started, full_content,
            tool_calls, tool_call_item_ids,
        ):
            yield ev

    if not completed:
        output_items = _build_final_output(
            msg_id, message_added, content_started, full_content,
            tool_calls, tool_call_item_ids,
        )
        yield _sse(
            "response.completed",
            {
                "id": resp_id,
                "object": "response",
                "created_at": now,
                "model": model,
                "status": "completed",
                "output": output_items,
                "usage": _convert_usage({}),
            },
        )


def _emit_close_events(
    msg_id: str,
    message_added: bool,
    content_started: bool,
    full_content: str,
    tool_calls: dict[int, dict],
    tool_call_item_ids: dict[int, str],
) -> list[str]:
    """生成关闭事件序列。"""
    events: list[str] = []

    # 关闭文本
    if content_started:
        events.append(
            _sse(
                "response.output_text.done",
                {
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": full_content,
                },
            )
        )
        events.append(
            _sse(
                "response.content_part.done",
                {
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": full_content, "annotations": []},
                },
            )
        )

    if message_added:
        events.append(
            _sse(
                "response.output_item.done",
                {
                    "output_index": 0,
                    "item": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": (
                            [{"type": "output_text", "text": full_content, "annotations": []}]
                            if content_started
                            else []
                        ),
                    },
                },
            )
        )

    # 关闭工具调用
    for idx in sorted(tool_calls):
        tc = tool_calls[idx]
        item_id = tool_call_item_ids[idx]
        output_idx = (1 if message_added else 0) + idx

        events.append(
            _sse(
                "response.function_call_arguments.done",
                {
                    "item_id": item_id,
                    "output_index": output_idx,
                    "arguments": tc["arguments"],
                },
            )
        )
        events.append(
            _sse(
                "response.output_item.done",
                {
                    "output_index": output_idx,
                    "item": {
                        "id": item_id,
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "status": "completed",
                    },
                },
            )
        )

    return events


def _build_final_output(
    msg_id: str,
    message_added: bool,
    content_started: bool,
    full_content: str,
    tool_calls: dict[int, dict],
    tool_call_item_ids: dict[int, str],
) -> list[dict]:
    """构建最终 output 数组，用于 response.completed 事件。"""
    output: list[dict] = []

    if message_added:
        output.append(
            {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": (
                    [{"type": "output_text", "text": full_content, "annotations": []}]
                    if content_started
                    else []
                ),
            }
        )

    for idx in sorted(tool_calls):
        tc = tool_calls[idx]
        output.append(
            {
                "id": tool_call_item_ids[idx],
                "type": "function_call",
                "call_id": tc["id"],
                "name": tc["name"],
                "arguments": tc["arguments"],
                "status": "completed",
            }
        )

    return output


# ─────────────────────── CC 非流式收集（用于 /v1/chat/completions 强制流式） ───────


async def collect_chat_completion(response) -> dict:
    """收集 CC stream chunk，拼装成标准 CC 非流式响应。"""
    full_content = ""
    tool_calls: dict[int, dict] = {}
    model = ""
    completion_id = ""
    system_fp = ""
    usage: dict = {}
    finish_reason = None

    async for chunk in parse_sse_stream(response):
        completion_id = chunk.get("id", completion_id)
        model = chunk.get("model", model)
        system_fp = chunk.get("system_fingerprint", system_fp)

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})

            if content := delta.get("content"):
                full_content += content

            if tc_list := delta.get("tool_calls"):
                for tc in tc_list:
                    idx = tc["index"]
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.get("id"):
                        tool_calls[idx]["id"] = tc["id"]
                    if name := tc.get("function", {}).get("name"):
                        tool_calls[idx]["function"]["name"] = name
                    if args := tc.get("function", {}).get("arguments"):
                        tool_calls[idx]["function"]["arguments"] += args

            if fr := choice.get("finish_reason"):
                finish_reason = fr

        if u := chunk.get("usage"):
            usage = u

    message: dict = {"role": "assistant", "content": full_content or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

    result: dict = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }
    if system_fp:
        result["system_fingerprint"] = system_fp

    return result


# ─────────────────────── usage 映射 ───────────────────────


def _convert_usage(u: dict) -> dict:
    if not u:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": u.get("prompt_tokens", 0),
        "output_tokens": u.get("completion_tokens", 0),
        "output_tokens_details": {
            "reasoning_tokens": u.get("completion_tokens_details", {}).get(
                "reasoning_tokens", 0
            )
        },
        "total_tokens": u.get("total_tokens", 0),
    }
