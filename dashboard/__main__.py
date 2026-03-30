"""Launch the Fallax dashboard server."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="reasonbench-dashboard",
        description="Launch the Fallax experiment dashboard",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd(),
        help="Parent directory containing experiment output directories",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8050, help="Bind port")
    args = parser.parse_args()

    import uvicorn

    from .api import create_app

    app = create_app(data_dir=args.data_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
