from pathlib import Path
from src.config import DATA_DIR
import pandas as pd


REQUIRED_FREQ_COLS = {"IDpol", "ClaimNb", "Exposure", "VehGas"}
REQUIRED_SEV_COLS = {"IDpol", "ClaimAmount"}

FILE_PAIRS = {
    "frequency": "freMTPL2freq.csv",
    "severity": "freMTPL2sev.csv",
}


def load_raw_data(data_dir : str | Path | None = None):
    data_dir = DATA_DIR if data_dir is None else Path(data_dir)

    freq_path = data_dir / FILE_PAIRS["frequency"]
    sev_path = data_dir / FILE_PAIRS["severity"]

    if not freq_path.exists():
        raise FileNotFoundError(f"Frequency data file not found: {freq_path}")
    if not sev_path.exists():
        raise FileNotFoundError(f"Severity data file not found: {sev_path}")

    freq_df = pd.read_csv(freq_path)
    sev_df = pd.read_csv(sev_path)

    if freq_df.empty or sev_df.empty:
        raise ValueError("Loaded dataset is empty")

    missing_freq = REQUIRED_FREQ_COLS.difference(freq_df.columns)
    missing_sev = REQUIRED_SEV_COLS.difference(sev_df.columns)
    if missing_freq:
        raise ValueError(
            f"Frequency dataset missing required columns: {sorted(missing_freq)}"
        )
    if missing_sev:
        raise ValueError(
            f"Severity dataset missing required columns: {sorted(missing_sev)}"
        )

    if not pd.api.types.is_numeric_dtype(freq_df["ClaimNb"]):
        raise ValueError("ClaimNb must be numeric")
    if not pd.api.types.is_numeric_dtype(freq_df["Exposure"]):
        raise ValueError("Exposure must be numeric")
    if not pd.api.types.is_numeric_dtype(sev_df["ClaimAmount"]):
        raise ValueError("ClaimAmount must be numeric")

    return freq_df, sev_df

