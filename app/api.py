import asyncio
from fastapi import Depends, FastAPI, HTTPException
from statistics import median
import time

from app.lifespan import lifespan
from app.schema import EventBatch
from app.state import app_state


app = FastAPI(lifespan=lifespan)


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
    # Use a single reference timestamp so all windows are evaluated against
    # the same cutoff while this handler runs.
    reference_ts = int(time.time())
    return {
        "total ingest requests": state.ingest_requests_total,
        "total events received": state.events_received_total,
        "last ingest time": state.ingest_last_ts,
        "model eval calls": state.inference_calls,
        "queue rejections": state.queue_rejections,
        "median of user medians": await state.median_of_medians(reference_ts),
    }
