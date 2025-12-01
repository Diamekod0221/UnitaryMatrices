from __future__ import annotations
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
GINIBRE_OUTPUT_DIR: Path = DATA_DIR /"ginibre_generation"

# Ensure it exists at import time
GINIBRE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PI_ESTIMATION_OUTPUT_DIR: Path = DATA_DIR / "pi_estimation"
