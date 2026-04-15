"""入口：python -m route_proxy"""

import uvicorn

uvicorn.run("route_proxy.app:app", host="127.0.0.1", port=8000)
