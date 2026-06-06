import pandas as pd


def preprocess_raw_data(freq_df: pd.DataFrame, sev_df: pd.DataFrame):
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


def preprocess_data_model(df:pd.DataFrame):
    df = df.copy()
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df["Frequency"] = df["Frequency"].clip(upper=20)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1.1)
    df.loc[(df["ClaimNb"] > 0) & (df["ClaimAmount"] == 0), "ClaimNb"] = 0

    return df
