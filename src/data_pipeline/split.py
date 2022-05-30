import pandas as pd
import numpy as np
import yaml

# might later by passed by argparse
file_in = "master.csv"
share_train = 0.6
share_test = 0.2

if __name__ == "__main__":

    with open("resources/paths.yaml") as f:
        dct_paths = yaml.safe_load(f)

    df = pd.read_csv(dct_paths["data"] + "/" + file_in)

    with open(dct_paths["log"] + "/" + file_in[:-4] + "_dtypes.yaml") as f:
        dct_dtypes = yaml.safe_load(f)

    df = df.astype(dct_dtypes)

    df = df.sort_values(["year_sold", "mo_sold"]).reset_index(drop=True)

    features = [i for i in df.columns if i != "sale_price"]

    ind_train = np.floor(len(df) * share_train)
    ind_test = np.floor(len(df) * (share_train + share_test))

    X_train = df.loc[:ind_train, features]
    y_train = df.loc[:ind_train, "sale_price"]

    X_test = df.loc[ind_train:ind_test, features]
    y_test = df.loc[ind_train:ind_test, "sale_price"]

    X_val = df.loc[ind_test:, features]
    y_val = df.loc[ind_test:, "sale_price"]

    X_train.to_csv(dct_paths["data"] + "/X_train.csv", index=False)
    y_train.to_csv(dct_paths["data"] + "/y_train.csv", index=False)

    X_test.to_csv(dct_paths["data"] + "/X_test.csv", index=False)
    y_test.to_csv(dct_paths["data"] + "/y_test.csv", index=False)

    X_val.to_csv(dct_paths["data"] + "/X_val.csv", index=False)
    y_val.to_csv(dct_paths["data"] + "/y_val.csv", index=False)
