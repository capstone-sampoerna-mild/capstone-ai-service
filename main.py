"""Single entrypoint.

This repo previously had two different FastAPI apps (`main.py` and `app/main.py`).
To avoid divergent behavior, keep a single app in `app/main.py` and re-export it here.
"""

from app.main import app

__all__ = ["app"]
