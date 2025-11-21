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
    """
    Return the global state
    """
    return app_state


@app.post("/ingest")
async def ingest(batch: EventBatch, state=Depends(get_state)):
    """
    Ingest an event and submit it to the event queue
    :param batch: a batch of events
    :param state: the global app state (injected by FastAPI)
    :return: an acknowledgement of reception of the event batch, or an error
    """
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
    """
    Get the rolling window median of results for a user
    :param user_id: the user id
    :param state: the global app state (injected by FastAPI)
    :return: a response with the median of the user's score,
        or an error if there's no data in the past five minutes
    """
    now = int(time.time())
    cutoff = now - state.window_seconds
    window = await state.trim_user_window(user_id, cutoff)

    if not window:
        raise HTTPException(status_code=404, detail=f"No data for {user_id}")

    scores = [score for (_, score) in window]
    return {"user_id": user_id, "median": median(scores)}


@app.get("/stats")
async def get_stats(state=Depends(get_state)):
    """
    Gets statistics to do with the state of the API
    :param state: the global app state (injected by FastAPI)
    :return:
        total ingest requests: the total number of calls to the ingestion POST endpoint
        total events received: total events received by the event queue
        last ingest time: the latest ingestion time
        model eval calls: the number of times the model was evaluated
        queue rejections: the number of times the queue rejected an event (because it was full)
        median of user medians: the median of all the medians for each user over the last five mins
    """
    reference_ts = int(time.time())
    return {
        "total ingest requests": state.ingest_requests_total,
        "total events received": state.events_received_total,
        "last ingest time": state.ingest_last_ts,
        "model eval calls": state.inference_calls,
        "queue rejections": state.queue_rejections,
        "median of user medians": await state.median_of_medians(reference_ts),
    }


# Profiling
if os.getenv("ENABLE_REQUEST_TIMING"):
    @app.middleware("http")
    async def timing_middleware(request, call_next):
        """
        Logs the time of a request on each HTTP request.
        """
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
