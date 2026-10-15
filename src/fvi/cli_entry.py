"""
Console Script Entry Points (`fvi-server` and `fvi-bench`) for `pyproject.toml`.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]


def run_web_server() -> None:
    server_script = ROOT_DIR / "app" / "server.py"
    sys.path.insert(0, str(ROOT_DIR / "src"))
    runpy.run_path(str(server_script), run_name="__main__")


def run_cli_benchmark() -> None:
    cli_script = ROOT_DIR / "scripts" / "run_fvi_cli.py"
    sys.path.insert(0, str(ROOT_DIR / "src"))
    runpy.run_path(str(cli_script), run_name="__main__")
