import pandas as pd
import pickle
import yaml
from datetime import datetime


if __name__ == "__main__":

    with open("resources/paths.yaml") as f:
        dct_paths = yaml.safe_load(f)

    with open(dct_paths["data"] + "/cat_values.yaml") as f:
        dct_cat = yaml.safe_load(f)

    with open(dct_paths["log"] + "/clean_dtypes.yaml") as f:
        dct_dtypes = yaml.safe_load(f)

    with open(
        dct_paths["experiment_results"]
        + "/"
        + dct_paths["latest_model"]
        + "/model.pkl",
        "rb",
    ) as f:
        model_final = pickle.load(f)

    df = pd.read_csv(dct_paths["data"] + "/mock_input.csv", dtype=dct_dtypes)

    df_out = pd.DataFrame({"y_hat": model_final.predict(df)})

    df_out.to_csv(dct_paths["data"] + "/mock_predictions.csv", index=False)
