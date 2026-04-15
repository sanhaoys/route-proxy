"""Responses API request -> Chat Completions API request."""

from __future__ import annotations


def convert_request(body: dict) -> dict:
    """将 Responses API 请求体转换为 Chat Completions 格式。"""
    result: dict = {"model": body["model"]}

    # ── messages ──
    messages: list[dict] = []

    if instructions := body.get("instructions"):
        messages.append({"role": "system", "content": instructions})

    input_val = body.get("input", "")
    if isinstance(input_val, str):
        messages.append({"role": "user", "content": input_val})
    elif isinstance(input_val, list):
        messages.extend(_convert_input_items(input_val))

    result["messages"] = messages

    # ── tools ──
    if tools := body.get("tools"):
        converted = [_convert_tool(t) for t in tools]
        result["tools"] = [t for t in converted if t is not None]
        if not result["tools"]:
            del result["tools"]

    # ── tool_choice / parallel_tool_calls ──
    if "tool_choice" in body:
        result["tool_choice"] = body["tool_choice"]
    if "parallel_tool_calls" in body:
        result["parallel_tool_calls"] = body["parallel_tool_calls"]

    # ── max_output_tokens -> max_completion_tokens ──
    if "max_output_tokens" in body:
        result["max_completion_tokens"] = body["max_output_tokens"]

    # ── text.format -> response_format ──
    if text := body.get("text"):
        if fmt := text.get("format"):
            result["response_format"] = _convert_response_format(fmt)

    # ── reasoning -> reasoning_effort ──
    if reasoning := body.get("reasoning"):
        if effort := reasoning.get("effort"):
            result["reasoning_effort"] = effort

    # ── 直接透传字段 ──
    for key in ("temperature", "top_p", "user", "store", "metadata", "seed"):
        if key in body:
            result[key] = body[key]

    return result


# ─────────────────────── input items ───────────────────────


def _convert_input_items(items: list[dict]) -> list[dict]:
    """将 Responses input 数组转为 Chat Completions messages。

    核心逻辑：将连续的 function_call item 合并到前一个 assistant
    message 的 tool_calls 字段中。
    """
    messages: list[dict] = []
    pending_tool_calls: list[dict] = []
    last_assistant_msg: dict | None = None

    for item in items:
        item_type = item.get("type")
        role = item.get("role")

        # ── function_call → 缓冲 ──
        if item_type == "function_call":
            pending_tool_calls.append(
                {
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                }
            )
            continue

        # 遇到非 function_call item 时，先 flush 缓冲
        if pending_tool_calls:
            _flush_tool_calls(messages, pending_tool_calls, last_assistant_msg)
            pending_tool_calls = []
            last_assistant_msg = None

        # ── function_call_output → tool message ──
        if item_type == "function_call_output":
            last_assistant_msg = None
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                }
            )
            continue

        # ── 普通消息 ──
        msg = _convert_message(item)
        messages.append(msg)
        last_assistant_msg = msg if msg["role"] == "assistant" else None

    # flush 尾部
    if pending_tool_calls:
        _flush_tool_calls(messages, pending_tool_calls, last_assistant_msg)

    return messages


def _flush_tool_calls(
    messages: list[dict],
    tool_calls: list[dict],
    last_assistant_msg: dict | None,
) -> None:
    """将缓冲的 tool_calls 合并到前一个 assistant 消息，或新建一个。"""
    if last_assistant_msg is not None:
        last_assistant_msg["tool_calls"] = tool_calls
    else:
        messages.append({"role": "assistant", "tool_calls": tool_calls})


# ─────────────────────── message ───────────────────────


def _convert_message(item: dict) -> dict:
    role = item.get("role", "user")
    content = item.get("content")

    if content is None or isinstance(content, str):
        return {"role": role, "content": content or ""}

    if isinstance(content, list):
        converted = [_convert_content_part(p) for p in content]
        # assistant 消息在 CC 中通常用纯文本
        if role == "assistant":
            text_parts = [
                p["text"] for p in converted if p.get("type") == "text" and p.get("text")
            ]
            return {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
        return {"role": role, "content": converted}

    return {"role": role, "content": content}


def _convert_content_part(part: dict) -> dict:
    t = part.get("type", "")

    if t in ("input_text", "output_text"):
        return {"type": "text", "text": part["text"]}

    if t == "input_image":
        url = part.get("image_url") or part.get("url", "")
        detail = part.get("detail", "auto")
        return {"type": "image_url", "image_url": {"url": url, "detail": detail}}

    if t == "input_audio":
        return {"type": "input_audio", "input_audio": part.get("audio", {})}

    # text / image_url 等已经是 CC 格式
    return part


# ─────────────────────── tools ───────────────────────


def _convert_tool(tool: dict) -> dict | None:
    if tool.get("type") != "function":
        # web_search / file_search / code_interpreter 等内置工具
        # Chat Completions 不支持，跳过
        return None

    func: dict = {"name": tool["name"]}
    if "description" in tool:
        func["description"] = tool["description"]
    if "parameters" in tool:
        func["parameters"] = tool["parameters"]
    if "strict" in tool:
        func["strict"] = tool["strict"]

    return {"type": "function", "function": func}


# ─────────────────────── response_format ───────────────────────


def _convert_response_format(fmt: dict) -> dict:
    fmt_type = fmt.get("type", "")

    if fmt_type == "json_object":
        return {"type": "json_object"}

    if fmt_type == "json_schema":
        schema_obj: dict = {"name": fmt.get("name", "response")}
        if "schema" in fmt:
            schema_obj["schema"] = fmt["schema"]
        if "strict" in fmt:
            schema_obj["strict"] = fmt["strict"]
        if "description" in fmt:
            schema_obj["description"] = fmt["description"]
        return {"type": "json_schema", "json_schema": schema_obj}

    if fmt_type == "text":
        return {"type": "text"}

    return fmt
