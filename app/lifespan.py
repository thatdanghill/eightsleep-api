import asyncio
import contextlib
from fastapi import FastAPI

from app.create_model import load_model
from app.state import app_state
from app.workers import inference_worker, persistence_worker


NUM_WORKERS = 4


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP ----
    model = load_model()
    app.state.model = model

    # Restore persisted state if available
    await app_state.load_from_file(app_state.state_file)

    # Create inference workers
    tasks = []
    for _ in range(NUM_WORKERS):
        task = asyncio.create_task(inference_worker(model, app_state))
        tasks.append(task)

    # Persist state periodically
    if app_state.state_file:
        tasks.append(
            asyncio.create_task(
                persistence_worker(app_state, app_state.state_file)
            )
        )

    app.state.worker_tasks = tasks

    yield

    # ---- SHUTDOWN ----
    for task in tasks:
        task.cancel()

    # Await cancellation to avoid warnings
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task
