from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointmass_notebook import main  # noqa: E402


def _forward_args(command: str) -> list[str]:
    """Preserve global CLI flags (e.g. --config) before command injection."""
    argv = sys.argv[1:]
    global_args: list[str] = []
    command_args: list[str] = []

    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--config":
            if i + 1 >= len(argv):
                raise SystemExit("error: argument --config: expected one argument")
            global_args.extend([token, argv[i + 1]])
            i += 2
            continue
        if token.startswith("--config="):
            global_args.append(token)
            i += 1
            continue
        command_args.append(token)
        i += 1

    return [*global_args, command, *command_args]


if __name__ == "__main__":
    main(_forward_args("train-sft"))

