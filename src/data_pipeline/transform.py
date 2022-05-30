import pandas as pd
import yaml
import datetime

# filenames may later be passed to the script via argparse
file_in = "clean.csv"
file_out = "master.csv"

if __name__ == "__main__":

    with open("resources/paths.yaml") as f:
        dct_paths = yaml.safe_load(f)

    with open(dct_paths["log"] + "/" + file_in[:-4] + "_dtypes.yaml") as f:
        dct_dtypes = yaml.safe_load(f)

    df = pd.read_csv(dct_paths["data"] + "/" + file_in, dtype=dct_dtypes)

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

    # one-hot encoding for categorical columns, drop original columns
    for col, dtype in df.dtypes.to_dict().items():
        if str(dtype) == "object":
            df_tmp = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, df_tmp], axis=1).drop(col, axis=1)

    df.to_csv(dct_paths["data"] + "/" + file_out, index=False)

    dct_dtypes = {key: str(value) for key, value in df.dtypes.to_dict().items()}

    with open(dct_paths["log"] + "/" + file_out[:-4] + "_dtypes.yaml", "w") as f:
        yaml.dump(dct_dtypes, f, default_flow_style=False)
