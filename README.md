# Route Proxy

解决 Hermes 对 GPT-5.x 模型强制使用 `/v1/responses` 请求，但上游不支持的问题。

代理接收 `/v1/responses` 请求，转换为 `/v1/chat/completions` 调用上游，响应转回 Responses 格式。所有请求内部强制 stream，按原始请求返回 stream/非stream。

## 使用

```bash
cp .env.example .env
# 编辑 .env 配置 BASE_URL
docker compose up --build
```

## 配置

```env
BASE_URL=https://your-upstream.com/api
HOST_PORT=8000
```

## 已知限制

- `previous_response_id` 不支持
- 内置工具（`web_search` 等）会被过滤
