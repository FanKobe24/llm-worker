"""
arq 任务：OpenAI 兼容 POST /chat/completions；结果写入消息指定的 Redis 列表键。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.parse import urlparse

import httpx

import config as worker_config

logger = logging.getLogger(__name__)


def _api_base_host_for_log(base_url: str) -> str | None:
    raw = (base_url or "").strip().rstrip("/")
    if not raw:
        return None
    try:
        pr = urlparse(raw if "://" in raw else f"https://{raw}")
        if pr.netloc:
            scheme = pr.scheme or "https"
            return f"{scheme}://{pr.netloc}"
        return raw
    except Exception:
        return raw


def _extract_message_content(data: dict[str, Any]) -> tuple[str | None, str | None]:
    """尽量兼容 OpenAI 风格 choices[0].message.content。"""
    try:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None, "empty_choices"
        ch0 = choices[0]
        if not isinstance(ch0, dict):
            return None, "bad_choice_shape"
        msg = ch0.get("message")
        if isinstance(msg, dict) and msg.get("content") is not None:
            return str(msg.get("content")), None
        if ch0.get("text") is not None:
            return str(ch0.get("text")), None
        return None, "no_message_content"
    except Exception as e:
        return None, str(e)


def _coerce_enqueued_at_ms(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _with_timings(
    base: dict[str, Any],
    *,
    enqueued_at_ms: int | None,
    processing_started_at_ms: int,
    processing_finished_at_ms: int,
) -> dict[str, Any]:
    out = dict(base)
    out["enqueued_at_ms"] = enqueued_at_ms
    out["processing_started_at_ms"] = processing_started_at_ms
    out["processing_finished_at_ms"] = processing_finished_at_ms
    return out


async def _push_result(
    redis: Any,
    *,
    out: dict[str, Any],
    rq: str,
    rt: str,
) -> None:
    raw = json.dumps(out, ensure_ascii=False)
    if rq:
        await redis.lpush(rq, raw)
    if rt:
        await redis.lpush(rt, raw)


async def llm_openai_chat(ctx: dict[str, Any], payload: dict[str, Any]) -> None:
    """
    payload:
      llm_task_id, base_url, model_id, api_key, messages, temperature,
      timeout_seconds, result_queue (optional), reply_to (optional),
      model_name (optional), extra_headers (optional dict),
      enqueued_at_ms (optional): 生产者将任务写入 arq 队列时的 Unix 毫秒时间
    """
    redis = ctx["redis"]
    processing_started_at_ms = int(time.time() * 1000)
    enqueued_at_ms = _coerce_enqueued_at_ms(payload.get("enqueued_at_ms"))

    llm_task_id = int(payload.get("llm_task_id") or 0)
    result_queue = payload.get("result_queue")
    rq = str(result_queue).strip() if result_queue else ""
    reply_to = payload.get("reply_to")
    rt = str(reply_to).strip() if reply_to else ""

    if not rq and not rt:
        logger.info(
            "核心日志: 消息未包含 result_queue / reply_to，跳过执行 llm_task_id=%s",
            llm_task_id,
        )
        return

    # result_queue：要求 Redis 中已有该输出列表键，或存在由生产者在入队前写入的占位键 {rq}:llm_dest（reply_to 同步通道不校验）
    if rq:
        dest_ready = await redis.exists(rq) > 0 or await redis.exists(f"{rq}:llm_dest") > 0
        if not dest_ready:
            logger.info(
                "核心日志: result_queue 在 Redis 中不存在且无 llm_dest 占位，跳过执行 key=%s llm_task_id=%s",
                rq,
                llm_task_id,
            )
            return

    base_raw = str(payload.get("base_url") or "").strip()
    base = worker_config.normalize_openai_compatible_base_url(base_raw)
    model_id = str(payload.get("model_id") or "").strip()
    api_key = payload.get("api_key")
    key_s = str(api_key).strip() if api_key is not None else ""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        messages = []
    temp = payload.get("temperature")
    if isinstance(temp, (int, float)):
        temperature = float(temp)
    else:
        temperature = worker_config.DEFAULT_TEMPERATURE
    raw_to = payload.get("timeout_seconds")
    timeout_s = float(raw_to) if raw_to is not None else worker_config.LLM_HTTP_TIMEOUT_SECONDS
    timeout_s = max(5.0, timeout_s)
    model_name = str(payload.get("model_name") or "").strip() or "unknown"
    extra = payload.get("extra_headers")
    if not isinstance(extra, dict):
        env_extra = worker_config.parse_extra_headers_json()
        extra = env_extra if env_extra else None
    host_log = _api_base_host_for_log(base)

    async def push_timed(err_dict: dict[str, Any]) -> None:
        fin = int(time.time() * 1000)
        await _push_result(
            redis,
            out=_with_timings(
                err_dict,
                enqueued_at_ms=enqueued_at_ms,
                processing_started_at_ms=processing_started_at_ms,
                processing_finished_at_ms=fin,
            ),
            rq=rq,
            rt=rt,
        )

    if not base or not model_id:
        await push_timed({"llm_task_id": llm_task_id, "ok": False, "error_message": "missing base_url or model_id"})
        return
    if not key_s:
        await push_timed(
            {
                "llm_task_id": llm_task_id,
                "ok": False,
                "error_message": "missing api_key",
                "model_id": model_id,
                "api_base_host": host_log,
                "model_name": model_name,
            }
        )
        return

    url = f"{base}/chat/completions"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {key_s}",
        "Content-Type": "application/json",
    }
    if isinstance(extra, dict):
        for k, v in extra.items():
            if str(k).strip():
                headers[str(k)] = str(v)
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
    }
    logger.info(
        "核心日志: worker LLM 请求开始 task_id=%s model=%s url host=%s",
        llm_task_id,
        model_id,
        host_log,
    )
    t0 = time.perf_counter()
    http_status: int | None = None
    try:
        client: httpx.AsyncClient = ctx["http"]
        r = await client.post(url, headers=headers, json=body, timeout=timeout_s)
        http_status = r.status_code
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError("response json not object")
        content, perr = _extract_message_content(data)
        if perr and not content:
            raise ValueError(perr)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        out = {
            "llm_task_id": llm_task_id,
            "ok": True,
            "content": content,
            "usage": usage,
            "duration_ms": elapsed_ms,
            "http_status": http_status,
            "error_message": None,
            "model_id": model_id,
            "api_base_host": host_log,
            "model_name": model_name,
        }
        logger.info(
            "核心日志: worker LLM 完成 task_id=%s 耗时_ms=%s 字符数=%s",
            llm_task_id,
            elapsed_ms,
            len(content or ""),
        )
    except httpx.HTTPStatusError as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        body_snip = ""
        try:
            body_snip = (e.response.text or "")[:2000]
        except Exception:
            pass
        msg = f"{e!s}"
        if body_snip:
            msg = f"{msg} body={body_snip}"
        http_status = e.response.status_code if e.response else http_status
        logger.warning("核心日志: worker LLM HTTP 错误 task_id=%s status=%s", llm_task_id, http_status)
        out = {
            "llm_task_id": llm_task_id,
            "ok": False,
            "content": None,
            "usage": None,
            "duration_ms": elapsed_ms,
            "http_status": http_status,
            "error_message": msg,
            "model_id": model_id,
            "api_base_host": host_log,
            "model_name": model_name,
        }
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.exception("核心日志: worker LLM 异常 task_id=%s", llm_task_id)
        out = {
            "llm_task_id": llm_task_id,
            "ok": False,
            "content": None,
            "usage": None,
            "duration_ms": elapsed_ms,
            "http_status": http_status,
            "error_message": str(e),
            "model_id": model_id,
            "api_base_host": host_log,
            "model_name": model_name,
        }

    fin = int(time.time() * 1000)
    await _push_result(
        redis,
        out=_with_timings(
            out,
            enqueued_at_ms=enqueued_at_ms,
            processing_started_at_ms=processing_started_at_ms,
            processing_finished_at_ms=fin,
        ),
        rq=rq,
        rt=rt,
    )
