from pathlib import Path
from src.config import DATA_DIR, features
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from src.data.loader import load_raw_data
from src.data.preprocessing import preprocess_data


def train_glm_baseline(data_dir : str | Path | None = None, test_size: float = 0.2, random_state: int = 42):

    data_dir = DATA_DIR if data_dir is None else Path(data_dir)

    freq_df, sev_df = load_raw_data(data_dir)
    df = preprocess_data(freq_df, sev_df)

    X = pd.get_dummies(df[features],dtype=float)
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
    y_pred = result.predict(X_test)

    return result, y_test, y_pred


if __name__ == "__main__":
    result, y_test, y_pred = train_glm_baseline()
    print("Poisson GLM trained successfully")
    print(f"Test actual sample: {y_test[:5].values}")
    print(f"Test predictions sample: {y_pred[:5].values}")
