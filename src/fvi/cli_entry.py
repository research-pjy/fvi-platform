from __future__ import annotations
from pathlib import Path
import runpy, sys

ROOT_DIR = Path(__file__).resolve().parents[2]

def run_web_server() -> None:
    sys.path.insert(0, str(ROOT_DIR / "src"))
    runpy.run_path(str(ROOT_DIR / "app" / "server.py"), run_name="__main__")

def run_cli_benchmark() -> None:
    sys.path.insert(0, str(ROOT_DIR / "src"))
    runpy.run_path(str(ROOT_DIR / "scripts" / "run_fvi_cli.py"), run_name="__main__")
