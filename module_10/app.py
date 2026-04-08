"""
Production LLM API — combines all Module 10 concepts.
"""
import os
import json
import time
import hashlib
import random
import logging
import sys
import asyncio
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-4o-mini"  # Same model for safety; swap to another in real usage
MAX_RETRIES = 2
BASE_DELAY = 0.5
MAX_DELAY = 8.0

# ──────────────────────────────────────────────
# STRUCTURED LOGGING (from mini-observability)
# ──────────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            entry.update(record.extra_data)
        return json.dumps(entry)

logger = logging.getLogger("llm_api")
logger.setLevel(logging.INFO)
logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# ──────────────────────────────────────────────
# COST TRACKING (from mini-cost)
# ──────────────────────────────────────────────
MODEL_PRICING = {
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "gpt-4o": {
        "input": 2.50 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
}

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    return prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]

# In-memory stats collector
stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "errors": 0,
    "total_cost_usd": 0.0,
    "total_tokens": 0,
    "started_at": datetime.utcnow().isoformat(),
}

# ──────────────────────────────────────────────
# CACHE (from mini-cache)
# ──────────────────────────────────────────────
response_cache: dict[str, dict] = {}

def cache_key(model: str, message: str, temperature: float) -> str:
    raw = f"{model}:{message}:{temperature}"
    return hashlib.sha256(raw.encode()).hexdigest()

# ──────────────────────────────────────────────
# RETRY WITH BACKOFF (from mini-retry)
# ──────────────────────────────────────────────
async def retry_with_backoff(coro_func, max_retries=MAX_RETRIES,
                              base_delay=BASE_DELAY, max_delay=MAX_DELAY):
    """
    Retry an async callable with exponential backoff + jitter.
    `coro_func` should be a zero-arg async callable (lambda or function).
    Returns (result, attempts).
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            result = await coro_func()
            return result, attempt + 1
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.5)
                logger.warning("Retry scheduled", extra={"extra_data": {
                    "attempt": attempt + 1, "delay": round(delay + jitter, 2),
                    "error": str(e)[:120],
                }})
                await asyncio.sleep(delay + jitter)
    raise last_exc

# ──────────────────────────────────────────────
# REQUEST / RESPONSE MODELS (from mini-fastapi)
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    use_cache: bool = Field(default=True)

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens: dict
    cost_usd: float
    latency_ms: float
    cached: bool

class ErrorResponse(BaseModel):
    error: str
    detail: str

# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────
app = FastAPI(title="Production LLM API", version="1.0.0")
aclient = AsyncOpenAI()


# ---------- HEALTH ----------
@app.get("/health")
async def health():
    return {"status": "healthy", "model": PRIMARY_MODEL}


# ---------- STATS (observability) ----------
@app.get("/stats")
async def get_stats():
    return {
        **stats,
        "cache_size": len(response_cache),
    }


# ---------- CHAT (main endpoint) ----------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    stats["total_requests"] += 1
    start = time.perf_counter()

    # --- Cache check (from mini-cache) ---
    key = cache_key(PRIMARY_MODEL, req.message, req.temperature)
    if req.use_cache and key in response_cache:
        stats["cache_hits"] += 1
        cached = response_cache[key]
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Cache hit", extra={"extra_data": {
            "latency_ms": round(latency_ms, 1), "cache_key": key[:12],
        }})
        return ChatResponse(
            response=cached["response"],
            model=cached["model"],
            tokens=cached["tokens"],
            cost_usd=0.0,  # Cached — no additional cost
            latency_ms=round(latency_ms, 1),
            cached=True,
        )

    # --- LLM call with retry + fallback (from mini-retry) ---
    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]
    last_error = None

    for model in models_to_try:
        try:
            async def call_llm():
                return await aclient.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": req.message}],
                    temperature=req.temperature,
                )

            resp, attempts = await retry_with_backoff(call_llm)
            break
        except Exception as e:
            last_error = e
            logger.error("Model failed", extra={"extra_data": {
                "model": model, "error": str(e)[:120],
            }})
    else:
        # All models failed — return error (from error-handling)
        stats["errors"] += 1
        raise HTTPException(status_code=503, detail=str(last_error)[:200])

    latency_ms = (time.perf_counter() - start) * 1000
    prompt_tokens = resp.usage.prompt_tokens
    completion_tokens = resp.usage.completion_tokens
    total_tokens = resp.usage.total_tokens
    cost = calculate_cost(model, prompt_tokens, completion_tokens)
    content = resp.choices[0].message.content

    # --- Update stats (observability + cost tracking) ---
    stats["total_cost_usd"] += cost
    stats["total_tokens"] += total_tokens

    # --- Store in cache ---
    response_cache[key] = {
        "response": content,
        "model": model,
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
    }

    # --- Structured log (from mini-observability) ---
    logger.info("Chat completed", extra={"extra_data": {
        "model": model, "attempts": attempts,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "cost_usd": round(cost, 8), "latency_ms": round(latency_ms, 1),
    }})

    return ChatResponse(
        response=content,
        model=model,
        tokens={"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens},
        cost_usd=round(cost, 8),
        latency_ms=round(latency_ms, 1),
        cached=False,
    )


# ---------- STREAMING CHAT (SSE) ----------
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    stats["total_requests"] += 1

    async def event_generator():
        try:
            stream = await aclient.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[{"role": "user", "content": req.message}],
                temperature=req.temperature,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield {"data": json.dumps({"token": delta})}
            yield {"data": json.dumps({"done": True})}
        except Exception as e:
            stats["errors"] += 1
            logger.error("Stream error", extra={"extra_data": {"error": str(e)[:120]}})
            yield {"data": json.dumps({"error": str(e)[:200]})}

    return EventSourceResponse(event_generator())


# ---------- GLOBAL ERROR HANDLER ----------
@app.exception_handler(Exception)
async def global_error_handler(request, exc):
    stats["errors"] += 1
    logger.error("Unhandled error", extra={"extra_data": {
        "error_type": type(exc).__name__, "error": str(exc)[:200],
    }})
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": "An unexpected error occurred."},
    )
