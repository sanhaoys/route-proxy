"""入口：python -m route_proxy"""

import uvicorn

uvicorn.run("route_proxy.app:app", host="0.0.0.0", port=8000)
