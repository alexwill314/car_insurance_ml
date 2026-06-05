from pathlib import Path
from src.config import DATA_DIR
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.load.loader import load_raw_data, preprocess_data

CATEGORICAL_COLS = ["Area", "VehGas", "Region", "VehBrand"]
NUMERIC_COLS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]


def _prepare_features(df: pd.DataFrame):
    X_num = df[NUMERIC_COLS].astype(float).reset_index(drop=True)
    X_cat = df[CATEGORICAL_COLS].astype(str).reset_index(drop=True)
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    cat_array = encoder.fit_transform(X_cat)
    cat_features = pd.DataFrame(
        cat_array,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=X_num.index,
    )
    features = pd.concat([X_num, cat_features], axis=1)
    return features


def train_glm_baseline(data_dir : str | Path | None = None, test_size: float = 0.2, random_state: int = 42):

    data_dir = DATA_DIR if data_dir is None else Path(data_dir)

    freq_df, sev_df = load_raw_data(data_dir)
    df = preprocess_data(freq_df, sev_df)

    X = _prepare_features(df)
    y = df["ClaimNb"].astype(float)
    offset = np.log(df["Exposure"].astype(float))

    X_train, X_test, y_train, y_test, offset_train, offset_test = train_test_split(
        X, y, offset, test_size=test_size, random_state=random_state
    )

    X_train = sm.add_constant(X_train, has_constant="add")
    X_test = sm.add_constant(X_test, has_constant="add")

    model = sm.GLM(
        y_train,
        X_train,
        family=sm.families.Poisson(),
        offset=offset_train,
    )
    result = model.fit()

    print(result.summary())

    return result, (X_train, y_train, offset_train), (X_test, y_test, offset_test)


if __name__ == "__main__":
    train_glm_baseline()
