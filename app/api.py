import asyncio
import logging
import os
from fastapi import Depends, FastAPI, HTTPException
from statistics import median
import time

from app.lifespan import lifespan
from app.schema import EventBatch
from app.state import app_state


app = FastAPI(lifespan=lifespan)
logger = logging.getLogger("app.api")


def get_state():
    return app_state


@app.post("/ingest")
async def ingest(batch: EventBatch, state=Depends(get_state)):
    accepted = 0
    for event in batch.events:
        try:
            await asyncio.wait_for(state.event_queue.put(event), timeout=0.05)
            accepted += 1
        except asyncio.TimeoutError:
            state.queue_rejections += 1
            raise HTTPException(503, detail="Queue full")

    await state.record_ingest(accepted)
    return {"status": "ok", "accepted": accepted, "batch_size": len(batch.events)}


@app.get("/users/{user_id}/median")
async def get_user_median(user_id: str, state=Depends(get_state)):
    now = int(time.time())
    cutoff = now - state.window_seconds
    window = await state.trim_user_window(user_id, cutoff)

    if not window:
        raise HTTPException(status_code=404, detail=f"No data for {user_id}")

    scores = [score for (_, score) in window]
    return {"user_id": user_id, "median": median(scores)}


@app.get("/stats")
async def get_stats(state=Depends(get_state)):
    reference_ts = int(time.time())
    return {
        "total ingest requests": state.ingest_requests_total,
        "total events received": state.events_received_total,
        "last ingest time": state.ingest_last_ts,
        "model eval calls": state.inference_calls,
        "queue rejections": state.queue_rejections,
        "median of user medians": await state.median_of_medians(reference_ts),
    }


if os.getenv("ENABLE_REQUEST_TIMING"):
    @app.middleware("http")
    async def timing_middleware(request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "request_timing",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response
