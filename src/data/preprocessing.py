import pandas as pd


def preprocess_data(freq_df: pd.DataFrame, sev_df: pd.DataFrame):
    freq_df = freq_df.copy()
    sev_df = sev_df.copy()

    sev_agg = sev_df.groupby("IDpol", as_index=False)["ClaimAmount"].sum()

    df = freq_df.merge(sev_agg, on="IDpol", how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].str.strip("'")
        df[col] = df[col].astype("category")

    return df
