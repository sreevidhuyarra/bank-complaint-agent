"""Hugging Face Spaces entry point.

Spaces runs `app.py` at the repo root; the UI itself lives in `app/app.py`.
"""

from app.app import main

if __name__ == "__main__":
    main()
