from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

LOAD_DIR = PROJECT_ROOT / "data"

MODELS_DIR = PROJECT_ROOT / "models"

features = [
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
    ]