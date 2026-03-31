from __future__ import annotations

import uvicorn

from api import app


def run() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
