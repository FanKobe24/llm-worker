"""
arq Worker 入口：arq run_worker.WorkerSettings
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from arq.connections import RedisSettings

import config as worker_config
from tasks import llm_openai_chat

logging.basicConfig(
    level=getattr(logging, worker_config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
    # 默认超时与单次 post(timeout=…)一致，避免连接层沿用 httpx 过短默认
    tout = httpx.Timeout(worker_config.LLM_HTTP_TIMEOUT_SECONDS)
    ctx["http"] = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max(32, worker_config.MAX_JOBS * 4), max_keepalive_connections=32),
        # HTTP/1.1：兼容各厂商 OpenAI 兼容网关（不少仅稳定测过 h1.1）；开启 h2 需 httpx[h2] 且部分反代对 h2 支持参差。
        # 本 worker 为短请求 POST JSON，keep-alive 下 h1.1 已足够。
        http2=False,
        timeout=tout,
    )
    logger.info(
        "核心日志: llm worker startup max_jobs=%s queue=%s",
        worker_config.MAX_JOBS,
        worker_config.ARQ_QUEUE_NAME,
    )


async def shutdown(ctx: dict[str, Any]) -> None:
    http: httpx.AsyncClient | None = ctx.get("http")
    if http is not None:
        await http.aclose()
    logger.info("核心日志: llm worker shutdown")


class WorkerSettings:
    logger.info(f"redis_dsn: {worker_config.get_redis_dsn()}")
    redis_settings = RedisSettings.from_dsn(worker_config.get_redis_dsn())
    functions = [llm_openai_chat]
    max_jobs = worker_config.MAX_JOBS
    queue_name = worker_config.ARQ_QUEUE_NAME
    on_startup = startup
    on_shutdown = shutdown
