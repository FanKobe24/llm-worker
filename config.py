"""
llm-worker 独立配置：仅读本目录相关环境变量（或容器 env），与主应用 .env 分离。
通用大模型调用进程，不包含业务语义。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env", encoding="utf-8-sig", override=True)
load_dotenv(_ROOT / ".env.local", encoding="utf-8-sig", override=True)


def _get(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key, default)
    return v if v is not None and v != "" else default


def _get_int(key: str, default: int) -> int:
    raw = _get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_secret(key: str, *aliases: str) -> str | None:
    for k in (key,) + aliases:
        raw = os.getenv(k)
        if raw is None:
            continue
        v = raw.strip()
        if v:
            return v
    return None


REDIS_URL: str = _get("REDIS_URL", "redis://127.0.0.1:6379/0") or "redis://127.0.0.1:6379/0"
REDIS_PASSWORD: str | None = _get_secret("REDIS_PASSWORD", "LLM_REDIS_PASSWORD")
# 与主站 enqueue 使用同一 arq 队列键（Redis ZSET）
ARQ_QUEUE_NAME: str = (_get("LLM_ARQ_QUEUE_NAME", "llm_task") or "llm_task").strip()
MAX_JOBS: int = max(1, _get_int("LLM_WORKER_MAX_JOBS", 4))
DEFAULT_TEMPERATURE: float = max(0.0, min(2.0, _get_float("LLM_DEFAULT_TEMPERATURE", 0.2)))
# 与主站一致：单次 OpenAI 兼容 POST 超时，默认 300s = 5 分钟（payload 未带 timeout_seconds 时用）
LLM_HTTP_TIMEOUT_SECONDS: float = max(5.0, _get_float("LLM_HTTP_TIMEOUT_SECONDS", 300.0))
LOG_LEVEL: str = (_get("LOG_LEVEL", "INFO") or "INFO").upper()

_CHAT_COMPLETIONS = "/chat/completions"


def _redis_dsn_apply_password(url: str, password: str | None) -> str:
    u = (url or "").strip()
    if not u or not password or not str(password).strip():
        return u
    pw = str(password).strip()
    p = urlparse(u)
    if p.password not in (None, ""):
        return u
    scheme = (p.scheme or "redis").strip() or "redis"
    host = p.hostname or "127.0.0.1"
    port_s = f":{p.port}" if p.port else ""
    path = p.path if p.path not in (None, "") else "/0"
    user = p.username
    if user:
        netloc = f"{quote(user, safe='')}:{quote(pw, safe='')}@{host}{port_s}"
    else:
        netloc = f":{quote(pw, safe='')}@{host}{port_s}"
    return urlunparse((scheme, netloc, path, "", "", ""))


def get_redis_dsn() -> str:
    """供 arq RedisSettings.from_dsn；合并 REDIS_PASSWORD 后的连接串。"""
    return _redis_dsn_apply_password(REDIS_URL, REDIS_PASSWORD)


def normalize_openai_compatible_base_url(raw: str) -> str:
    u = (raw or "").strip().rstrip("/")
    while u.lower().endswith(_CHAT_COMPLETIONS):
        u = u[: -len(_CHAT_COMPLETIONS)].rstrip("/")
    return u


def parse_extra_headers_json() -> dict[str, str] | None:
    raw = _get("LLM_EXTRA_HEADERS_JSON", "")
    if not raw or not str(raw).strip():
        return None
    try:
        obj = json.loads(str(raw).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out: dict[str, str] = {}
    for k, v in obj.items():
        out[str(k)] = str(v)
    return out or None
