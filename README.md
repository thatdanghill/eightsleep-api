# Eight Sleep Interview API

A small FastAPI app that ingests events, runs a PyTorch model for scoring, keeps per-user sliding windows, and exposes medians (per-user and a median-of-medians) over a rolling window.

## Running with Docker

Prerequisites: Docker 20.10+.

Build the image:
```sh
docker build -t eightsleep-api .
```

Run the API (port 8000):
```sh
docker run --rm -p 8000:8000 \
  -v data:/data \
  eightsleep-api
```

Uvicorn runs `app.api:app` with a single worker to keep the in-memory state (`user_windows`) consistent. The inference workers inside the app will spawn automatically at startup.

### State persistence

- The app will load persisted state on startup if `STATE_FILE_PATH` exists (defaults to `/data/state.json`).
- A background task saves state to that path every 15 seconds using an atomic write (tmp + replace).
- Mount a volume at `/data` to keep state across restarts: `-v eightsleep-state:/data`.

## Endpoints

- `POST /ingest` — ingest a batch of events.
- `GET /users/{user_id}/median` — median score for a single user within the rolling window.
- `GET /stats` — service stats plus `median of user medians` across all users.

Example ingest:
```sh
ts=$(date +%s)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d "{\"events\":[{\"user_id\":\"user-1\",\"timestamp\":$ts,\"features\":[0.1,0.2,0.3]}]}"
```

## JSON event generation

`event_generator.py` can fire synthetic traffic:
```sh
python event_generator.py
```

## Profiling

- Request timing middleware: set `ENABLE_REQUEST_TIMING=1` to log per-request durations (ms).
- CPU profiling: run through `profiling.py` to collect a cProfile file:
  ```sh
  PROFILE_OUTPUT=/tmp/profile.out python profiling.py
  # In another shell, generate JSON requests, then Ctrl+C when done.
  python -m pstats /tmp/profile.out
  # inside pstats:
  # stats.sort_stats("tottime").print_stats(30)
  ```
