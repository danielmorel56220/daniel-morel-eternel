"""Chemins du projet — utilisés par les scripts pipeline."""

from pathlib import Path

# pipeline/paths.py → racine du repo
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
TEXTES_DIR = ROOT / "textes_extraits"
BASE_VECTORIELLE = ROOT / "base_vectorielle"
LOGS_DIR = ROOT / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
