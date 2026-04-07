"""Convenience ASGI entrypoint.

Run from repo root:
    uvicorn main:app --reload

(Alternatively: uvicorn app.main:app --reload)
"""

from app.main import app

__all__ = ["app"]
