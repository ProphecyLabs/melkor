import pandas as pd
import json
import os


def to_snake(str_in, scream=False):
    """Convert string to snake_case or SNAKE_CASE"""
    # TODO: extend functionality to other characters that might need to be eliminated or replaced

    str_in = str_in.strip().replace(" ", "_").replace(",", "_").replace(":", "")

    if scream == False:
        return str_in.lower()
    else:
        return str_in.upper()


if __name__ == "__main__":

    with open("resources/paths.json", "r") as f:
        dct_paths = json.load(f)

    df = pd.read_csv(dct_paths["data"]["raw"])

    df.columns = [to_snake(col) for col in df.columns]

    df = df.drop(["unnamed_0", "misc_val"], axis=1)

    # define list of categorical columns
    cat_cols = [col for col in df.columns if str(df.dtypes[col]) == "object"]

    # convert strings in categorical columns to SNAKE_CASE
    for col in cat_cols:
        df[col] = df[col].apply(lambda x: to_snake(x, scream=True))

    # drop duplicate entries based on latitude and longitude columns
    df = (
        df.sort_values(["year_sold", "mo_sold"], ascending=False)
        .drop_duplicates(subset=["latitude", "longitude"])
        .reset_index(drop=True)
    )

    dct_dtypes = {key: str(value) for key, value in df.dtypes.to_dict().items()}

    for col in [
        "bsmt_full_bath",
        "bsmt_half_bath",
        "full_bath",
        "half_bath",
        "kitchen_abvgr",
        "fireplaces",
    ]:
        dct_dtypes[col] = "int64"

    df = df.astype(dct_dtypes)

    df.to_csv(dct_paths["data"]["clean"], index=False)

    # create log directory if it does not exist yet
    try:
        os.makedirs("log")
    except FileExistsError:
        pass

    with open(dct_paths["log"]["dtypes_clean"], "w") as f:
        json.dump(dct_dtypes, f, indent=4)
