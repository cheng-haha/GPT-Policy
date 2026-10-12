"""One isolated host process per queue attempt."""

import json
import signal
import sys
from pathlib import Path


def interrupted(*_):
    raise KeyboardInterrupt("Queue worker terminated")


def main():
    signal.signal(signal.SIGTERM, interrupted)
    job = json.loads(Path(sys.argv[1]).read_text())
    if job["smoke"]:
        from .smoke import run_smoke as execute
    else:
        from .runner import run_episode as execute
    result = execute(job["config"], job["evidence"], job["output"], job["gpu"])
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
