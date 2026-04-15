FROM python:3.13-slim

WORKDIR /app

# 先装依赖（利用 Docker 层缓存）
COPY pyproject.toml .
RUN pip install --no-cache-dir $(python3 -c "\
import tomllib; \
f = open('pyproject.toml', 'rb'); \
print(' '.join(tomllib.load(f)['project']['dependencies'])); \
f.close()")

# 再拷源码
COPY route_proxy/ route_proxy/

EXPOSE 8000

CMD ["python", "-m", "route_proxy"]
