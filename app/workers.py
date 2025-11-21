import asyncio
import time
from pathlib import Path
from typing import Callable

import torch
import numpy as np

from app.state import AppState
from app.schema import Event


async def inference_worker(
    model: torch.nn.Module,
    state: AppState,
    device: str = "cpu",
) -> None:
    """
    Continuously pull events from the queue, run inference, and
    update per-user rolling windows.
    """
    model.to(device)
    model.eval()

    while True:
        event: Event = await state.event_queue.get()

        # Run model inference on this event's features
        with torch.no_grad():
            x = torch.tensor(
                event.features,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)  # shape [1, in_dim]
            score_tensor = model(x)
            # model forward already squeezes last dim in provided script
            score = float(score_tensor.item())

        state.inference_calls += 1

        # Update per-user rolling window
        await state.insert_user_window(event.user_id, event.timestamp, score)

        # Drop anything older than window_seconds
        cutoff = event.timestamp - state.window_seconds
        await state.trim_user_window(event.user_id, cutoff)

        state.event_queue.task_done()


async def persistence_worker(
    state: AppState,
    path: Path,
    interval_seconds: int = 15,
) -> None:
    """
    Periodically persist in-memory state to disk.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        if not path:
            continue

        try:
            await state.save_to_file(path)
        except Exception:
            # Avoid crashing the loop; could replace with structured logging.
            pass
