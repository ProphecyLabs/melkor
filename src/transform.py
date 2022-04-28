import pandas as pd
import pickle
import datetime

if __name__ == "__main__":

    df = pd.read_csv("resources/df_clean.csv")

    # transform misc_feature values into dummies
    df_tmp = pd.get_dummies(df["misc_feature"])
    df_tmp = df_tmp.drop(["NONE", "OTHR"], axis=1)
    df_tmp.columns = ["elevator", "second_garage", "shed", "tennis_court"]
    df = pd.concat([df, df_tmp], axis=1)

    # age of building
    df["building_age"] = datetime.datetime.now().year - df.year_built

    # time since last renovation
    df["remod_age"] = df.year_remod_add - datetime.datetime.now().year

    # indicator whether building has been remodeled at all
    df["remod"] = df.apply(
        lambda row: 0 if row["year_built"] == row["year_remod_add"] else 1, axis=1
    )

    # indicator whether house has a second floor
    df["second_flr"] = df.second_flr_sf.apply(lambda x: 0 if x == 0 else 1)

    # indicator whether house has masonry veneer
    df["mas_vnr"] = df.mas_vnr_area.apply(lambda x: 0 if x == 0 else 1)

    # one-hot encoding for categorical columns, drop original columns
    for col, dtype in df.dtypes.to_dict().items():
        if str(dtype) == "object":
            df_tmp = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, df_tmp], axis=1).drop(col, axis=1)

    # save master dataset to csv
    df.to_csv("resources/master.csv", index=False)

    # generate dictionary of dtypes
    dct_dtypes = {key: str(value) for key, value in df.dtypes.to_dict().items()}
