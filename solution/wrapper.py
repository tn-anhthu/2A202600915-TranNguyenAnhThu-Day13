"""YOUR mitigation + observability layer. The simulator calls mitigate() around the
opaque agent (a REAL LLM) for every request. This is the ONLY place observability can
live -- the agent is silent. Legal moves: retry / cache / route / guardrail / sanitize
/ fallback / session-reset / PROMPT ROUTING, plus your own logging/tracing/metrics.
Illegal: hardcoding answers, importing the agent internals, reading instructor files,
network exfiltration.

  call_next(question, config) -> result   # the only way to reach the black box
  context = {"session_id","turn_index","qid","cache": <shared dict>, "cache_lock": <Lock>}
  result  = {"answer","status","steps","trace","meta":{latency_ms,usage,...}}
"""
from __future__ import annotations
import os
import sys
import time

from telemetry.logger import logger
from telemetry.cost import cost_from_usage
from telemetry.redact import redact

# ── inject deps/ so binary's Python 3.12 can find openai ─────────────────────
_deps = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deps")
if os.path.isdir(_deps) and _deps not in sys.path:
    sys.path.insert(0, _deps)

_stdlib312 = "/opt/anaconda3/envs/py312/lib/python3.12"
if os.path.isdir(_stdlib312) and _stdlib312 not in sys.path:
    sys.path.append(_stdlib312)

# ── fix Authorization header: binary sends "Bearer ollama" for local provider,
#    replace with the real API key from OPENAI_API_KEY env var ────────────────
try:
    import httpx as _httpx
    _api_key = os.environ.get("OPENAI_API_KEY", "")
    _orig_send = _httpx.Client.send

    def _patched_send(self, request, *a, **kw):
        if _api_key and request.headers.get("authorization") in ("Bearer ollama", "Bearer "):
            request.headers["authorization"] = f"Bearer {_api_key}"
        return _orig_send(self, request, *a, **kw)

    _httpx.Client.send = _patched_send
except Exception:
    pass


def mitigate(call_next, question, config, context):
    qid = context.get("qid", "?")
    turn = context.get("turn_index", 0)
    session = context.get("session_id", "?")

    last_result = None
    for attempt in range(1, 4):
        t0 = time.time()
        result = call_next(question, config)
        wall_ms = int((time.time() - t0) * 1000)

        meta = result.get("meta", {})
        usage = meta.get("usage", {})
        status = result.get("status", "")
        answer = result.get("answer") or ""

        logger.log_event("CALL", {
            "qid": qid,
            "session": session,
            "turn": turn,
            "attempt": attempt,
            "status": status,
            "wall_ms": wall_ms,
            "latency_ms": meta.get("latency_ms"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cost_usd": cost_from_usage(meta.get("model", ""), usage),
            "steps": result.get("steps", 0),
            "tools_used": meta.get("tools_used", []),
            "model": meta.get("model"),
            "pii_in_answer": redact(answer)[1] > 0,
        })

        # redact PII from answer before returning
        clean_answer, pii_count = redact(answer)
        if answer and pii_count > 0:
            result = dict(result)
            result["answer"] = clean_answer

        last_result = result

        if status == "ok" or attempt == 3:
            break

        logger.log_event("RETRY", {"qid": qid, "attempt": attempt, "status": status})

    return last_result
