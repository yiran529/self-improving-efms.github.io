from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointmass_notebook import main  # noqa: E402


if __name__ == "__main__":
    main(["train-self-improve", *sys.argv[1:]])

