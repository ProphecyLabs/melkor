import pandas as pd
import numpy as np
import yaml
import pickle

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as MSE

if __name__ == "__main__":

    with open("resources/paths.yaml") as f:
        dct_paths = yaml.safe_load(f)

    with open(
        dct_paths["experiment_results"]
        + "/"
        + dct_paths["latest_model"]
        + "/model.pkl",
        "rb",
    ) as f:
        model_final = pickle.load(f)

    with open(dct_paths["log"] + "/master_dtypes.yaml") as f:
        dct_dtypes = yaml.safe_load(f)

    X_train = pd.read_csv(dct_paths["data"] + "/X_train.csv", dtype=dct_dtypes)
    y_train = pd.read_csv(dct_paths["data"] + "/y_train.csv", dtype=dct_dtypes)

    X_test = pd.read_csv(dct_paths["data"] + "/X_test.csv", dtype=dct_dtypes)
    y_test = pd.read_csv(dct_paths["data"] + "/y_test.csv", dtype=dct_dtypes)

    X_train = pd.concat([X_train, X_test])
    y_train = pd.concat([y_train, y_test])

    X_val = pd.read_csv(dct_paths["data"] + "/X_val.csv", dtype=dct_dtypes)
    y_val = pd.read_csv(dct_paths["data"] + "/y_val.csv", dtype=dct_dtypes)

    # model uses default alpha value of 1
    model_base = Ridge(solver="cholesky")

    model_base.fit(X_train, y_train)

    y_hat = model_final.predict(X_val)

    rmse_mod = MSE(y_val, y_hat, squared=False)

    pred_base = model_base.predict(X_val)

    rmse_base = MSE(y_val, pred_base, squared=False)

    pred_med = np.median(y_train)

    rmse_med = MSE(y_val, np.repeat(pred_med, len(y_val)), squared=False)

    pred_mean = np.mean(y_train)

    rmse_mean = MSE(y_val, np.repeat(pred_mean, len(y_val)), squared=False)

    with open(dct_paths["models"] + "/latest.pkl", "wb") as f:
        pickle.dump(model_final, f)

    with open(dct_paths["models"] + "/baseline.pkl", "wb") as f:
        pickle.dump(model_base, f)

    with open(dct_paths["models"] + "/median.pkl", "wb") as f:
        pickle.dump(pred_med, f)

    with open(dct_paths["models"] + "/mean.pkl", "wb") as f:
        pickle.dump(pred_mean, f)
