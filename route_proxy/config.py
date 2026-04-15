"""集中管理所有环境变量。"""

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com").rstrip("/")

# 参考 OpenAI SDK 默认 600s read timeout
TIMEOUT = httpx.Timeout(connect=10, read=600, write=10, pool=10)

CC_URL = f"{BASE_URL}/v1/chat/completions"
