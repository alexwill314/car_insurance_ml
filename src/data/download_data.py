from src.config import DATA_DIR
from sklearn.datasets import fetch_openml
import os


def download_data():
    data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    print("Downloading frequency dataset (OpenML ID 41214)...")
    freq = fetch_openml(data_id=41214, as_frame=True)
    freq_path = os.path.join(data_dir, "freMTPL2freq.csv")
    freq.data.to_csv(freq_path, index=False)
    print(f"Saved frequency dataset to {freq_path}")

    print("Downloading severity dataset (OpenML ID 41215)...")
    sev = fetch_openml(data_id=41215, as_frame=True)
    sev_path = os.path.join(data_dir, "freMTPL2sev.csv")
    sev.data.to_csv(sev_path, index=False)
    print(f"Saved severity dataset to {sev_path}")


if __name__ == "__main__":
    download_data()
