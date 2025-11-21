"""
Profiler to run the app under cProfile.

Usage:
    PROFILE_OUTPUT=/tmp/profile.out python profiling.py
Inspect with:
    python -m pstats /tmp/profile.out
        pstats> stats.sort_stats("tottime").print_stats(30)
"""

import cProfile
import os
import pstats

import uvicorn


def main():
    output = os.getenv("PROFILE_OUTPUT", "/tmp/profile.out")
    profiler = cProfile.Profile()

    try:
        profiler.enable()
        uvicorn.run(
            "app.api:app",
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            reload=False,
            workers=1,
        )
    finally:
        profiler.disable()
        profiler.dump_stats(output)
        stats = pstats.Stats(profiler)
        stats.sort_stats("tottime").print_stats(20)
        print(f"\nProfile written to {output}")


if __name__ == "__main__":
    main()
