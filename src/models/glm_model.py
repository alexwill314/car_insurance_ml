from pathlib import Path
from src.config import DATA_DIR, features
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from src.data.loader import load_raw_data
from src.data.preprocessing import preprocess_raw_data, preprocess_data_model


def train_glm_baseline(data_dir : str | Path | None = None, test_size: float = 0.2, random_state: int = 42):

    data_dir = DATA_DIR if data_dir is None else Path(data_dir)

    freq_df, sev_df = load_raw_data(data_dir)
    df = preprocess_raw_data(freq_df, sev_df)
    df = preprocess_data_model(df)

    X = pd.get_dummies(df[features],dtype=float)
    y = df["ClaimNb"].astype(float)
    exposure = df["Exposure"].astype(float)

    X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(
        X, y, exposure, test_size=test_size, random_state=random_state
    )

    X_train = sm.add_constant(X_train, has_constant="add")
    X_test = sm.add_constant(X_test, has_constant="add")

    model = sm.GLM(
        y_train,
        X_train,
        family=sm.families.Poisson(),
        exposure=exposure_train,
    )
    result = model.fit()
    y_pred_freq = result.predict(X_test, exposure=exposure_test) / exposure_test
    y_test_freq = y_test / exposure_test

    return result, y_test_freq, y_pred_freq, exposure_test


if __name__ == "__main__":
    result, y_test_freq, y_pred_freq, exposure_test = train_glm_baseline()
