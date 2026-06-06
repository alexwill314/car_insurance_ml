import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, features
from src.data.loader import load_raw_data
from src.data.preprocessing import preprocess_raw_data, preprocess_data_model


def train_random_forest(data_dir=None,
                        test_size=0.2,
                        random_state=42,
                        n_estimators=100,
                        min_samples_leaf=20,
                        max_depth=8):
    if data_dir is None:
        data_dir = str(DATA_DIR)
    freq_df, sev_df = load_raw_data(data_dir)
    df = preprocess_raw_data(freq_df, sev_df)
    df = preprocess_data_model(df)
    X = pd.get_dummies(df[features])
    y = df["ClaimNb"] / df["Exposure"]
    w = df["Exposure"]

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        w,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        criterion="poisson",
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)

    return model, y_test, y_pred


if __name__ == "__main__":
    model, y_test, y_pred = train_random_forest()
    print("Random Forest trained successfully")
    print(f"Test actual sample (frequency): {y_test[:5].values}")
    print(f"Test predictions sample (frequency): {y_pred[:5]}")