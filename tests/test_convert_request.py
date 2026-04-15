"""测试 Responses API -> Chat Completions 请求转换。"""

from route_proxy.convert_request import convert_request


def test_simple_string_input():
    body = {"model": "gpt-4o", "input": "hello"}
    result = convert_request(body)

    assert result["model"] == "gpt-4o"
    assert result["messages"] == [{"role": "user", "content": "hello"}]


def test_instructions_become_system_message():
    body = {
        "model": "gpt-4o",
        "instructions": "You are helpful.",
        "input": "hi",
    }
    result = convert_request(body)

    assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert result["messages"][1] == {"role": "user", "content": "hi"}


def test_input_message_array():
    body = {
        "model": "gpt-4o",
        "input": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ],
    }
    result = convert_request(body)
    msgs = result["messages"]

    assert len(msgs) == 3
    assert msgs[0] == {"role": "user", "content": "Q1"}
    assert msgs[1] == {"role": "assistant", "content": "A1"}
    assert msgs[2] == {"role": "user", "content": "Q2"}


def test_content_type_conversion():
    """input_text / output_text -> text"""
    body = {
        "model": "gpt-4o",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ],
    }
    result = convert_request(body)
    content = result["messages"][0]["content"]

    assert content == [{"type": "text", "text": "hello"}]


def test_image_content_conversion():
    body = {
        "model": "gpt-4o",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "https://example.com/img.png"},
                ],
            },
        ],
    }
    result = convert_request(body)
    part = result["messages"][0]["content"][0]

    assert part["type"] == "image_url"
    assert part["image_url"]["url"] == "https://example.com/img.png"


def test_function_call_grouping():
    """连续的 function_call 应合并到同一个 assistant message。"""
    body = {
        "model": "gpt-4o",
        "input": [
            {"role": "user", "content": "Do things"},
            {"type": "function_call", "call_id": "c1", "name": "fn1", "arguments": "{}"},
            {"type": "function_call", "call_id": "c2", "name": "fn2", "arguments": '{"x":1}'},
            {"type": "function_call_output", "call_id": "c1", "output": "r1"},
            {"type": "function_call_output", "call_id": "c2", "output": "r2"},
        ],
    }
    result = convert_request(body)
    msgs = result["messages"]

    # user, assistant(tool_calls), tool, tool
    assert len(msgs) == 4
    assert msgs[1]["role"] == "assistant"
    assert len(msgs[1]["tool_calls"]) == 2
    assert msgs[1]["tool_calls"][0]["id"] == "c1"
    assert msgs[1]["tool_calls"][1]["function"]["name"] == "fn2"
    assert msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "r1"}
    assert msgs[3] == {"role": "tool", "tool_call_id": "c2", "content": "r2"}


def test_function_call_merges_into_preceding_assistant():
    """function_call 应合并到前面的 assistant 消息中。"""
    body = {
        "model": "gpt-4o",
        "input": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Let me check."},
            {"type": "function_call", "call_id": "c1", "name": "search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "found"},
        ],
    }
    result = convert_request(body)
    msgs = result["messages"]

    # user, assistant(content + tool_calls), tool
    assert len(msgs) == 3
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Let me check."
    assert msgs[1]["tool_calls"][0]["id"] == "c1"


def test_tool_conversion():
    body = {
        "model": "gpt-4o",
        "input": "hi",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"loc": {"type": "string"}}},
                "strict": True,
            },
            {"type": "web_search"},  # 内置工具，应被过滤
        ],
    }
    result = convert_request(body)

    assert len(result["tools"]) == 1
    tool = result["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["strict"] is True


def test_field_mappings():
    body = {
        "model": "gpt-4o",
        "input": "hi",
        "max_output_tokens": 500,
        "temperature": 0.5,
        "top_p": 0.9,
        "reasoning": {"effort": "high"},
        "text": {"format": {"type": "json_object"}},
    }
    result = convert_request(body)

    assert result["max_completion_tokens"] == 500
    assert "max_output_tokens" not in result
    assert result["temperature"] == 0.5
    assert result["top_p"] == 0.9
    assert result["reasoning_effort"] == "high"
    assert result["response_format"] == {"type": "json_object"}


def test_json_schema_format():
    body = {
        "model": "gpt-4o",
        "input": "hi",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "product",
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                "strict": True,
            }
        },
    }
    result = convert_request(body)
    rf = result["response_format"]

    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "product"
    assert rf["json_schema"]["strict"] is True
    assert "properties" in rf["json_schema"]["schema"]
