#!/usr/bin/env python3
"""Start the web application.

用法: python scripts/run_server.py --host 127.0.0.1 --port 8000

Launches the FastAPI application ``maritime_route.web.app:app`` under the
Uvicorn ASGI server. Host and port default to ``config.SERVER``; ``--reload``
restarts the server when a source file changes (development only).
The page is then available at ``http://<host>:<port>/``.

启动 Web 服务：使用 Uvicorn 运行 FastAPI 应用；--reload 仅用于开发调试。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the package importable without installation: add <project>/src to the
# module search path (parents[1] of scripts/run_server.py is the project root).
# 将 src 目录加入模块搜索路径，无需安装即可导入 maritime_route。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from maritime_route.config import SERVER


def main() -> int:
    """Parse the command line and run the server until it is stopped.

    Returns
    -------
    int
        Process exit code, 0 on a normal shutdown (Ctrl+C).
    """
    parser = argparse.ArgumentParser(description="Run the Maritime Route Planner server")
    parser.add_argument("--host", default=SERVER.host)
    parser.add_argument("--port", type=int, default=SERVER.port)
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"\n  Maritime Route Planner -> http://{args.host}:{args.port}\n")
    # The app is given as an import string (not an object) because Uvicorn's
    # reload mode must re-import it in a fresh worker process.
    uvicorn.run("maritime_route.web.app:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
