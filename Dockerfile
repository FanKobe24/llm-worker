FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY config.py tasks.py run_worker.py /app/

# 仅 Worker：通过环境变量注入 REDIS_URL 等（不在镜像内置 .env）
CMD ["arq", "run_worker.WorkerSettings"]
