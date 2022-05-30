import pandas as pd
import numpy as np
import random
import yaml
import datetime

file_in = "clean.csv"
file_out = "mock_input.csv"
len_out = 5  # number of rows of output dataframe

if __name__ == "__main__":

    with open("resources/paths.yaml") as f:
        dct_paths = yaml.safe_load(f)

    df = pd.read_csv(dct_paths["data"] + "/" + file_in)

    with open(dct_paths["log"] + "/" + file_in[:-4] + "_dtypes.yaml") as f:
        dct_dtypes = yaml.safe_load(f)

    df = df.astype(dct_dtypes)

    df["building_age"] = datetime.datetime.now().year - df.year_built

    df["remod_age"] = df.year_remod_add - datetime.datetime.now().year

    # indicator whether building has been remodeled at all
    df["remod"] = df.apply(
        lambda row: 0 if row["year_built"] == row["year_remod_add"] else 1, axis=1
    )

    # indicator whether house has a second floor
    df["second_flr"] = df.second_flr_sf.apply(lambda x: 0 if x == 0 else 1)

    # indicator whether house has masonry veneer
    df["mas_vnr"] = df.mas_vnr_area.apply(lambda x: 0 if x == 0 else 1)

    dct_out = {}

    for col in df.columns:
        dct_out[col] = random.choices(df[col].unique(), k=len_out)

    df_out = pd.DataFrame(dct_out)

    # one-hot encoding for categorical columns, drop original columns
    for col, dtype in df_out.dtypes.to_dict().items():
        if str(dtype) == "object":
            df_tmp = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, df_tmp], axis=1).drop(col, axis=1)

    # fill missing values from one-hot encoding
    ls_df = [df_out]
    for col in df.columns:
        if col not in df_out.columns:
            ls_df.append(pd.DataFrame({col: np.repeat(0, len_out)}))

    df_out = pd.concat(ls_df, axis=1)

    df_out = df_out[[i for i in df.columns if i != "sale_price"]]

    df_out.to_csv(dct_paths["data"] + "/" + file_out, index=False)
