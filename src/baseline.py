import pandas as pd
import numpy as np
import pickle
import json

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

rndseed = 42

if __name__ == "__main__":

    with open("resources/paths.json", "r") as f:
        dct_paths = json.load(f)

    df = pd.read_csv(dct_paths["master"])

    with open(dct_paths["log"]["dtypes_master"], "r") as f:
        dct_dtypes = json.load(f)

    df = df.astype(dct_dtypes)

    y = df.sale_price

    X = df[[i for i in df.columns if i != "sale_price"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=rndseed
    )

    # model uses default alpha value of 1
    model = Ridge(solver="cholesky")

    model.fit(X, y)

    y_hat = model.predict(X_test)

    rmse_model = mean_squared_error(y_test, y_hat, squared=False)

    pred_med = np.median(y_train)

    rmse_med = mean_squared_error(
        y_test, np.repeat(pred_med, len(y_test)), squared=False
    )

    pred_mean = np.mean(y_train)

    rmse_mean = mean_squared_error(
        y_test, np.repeat(pred_mean, len(y_test)), squared=False
    )

    dct_model = {"prediction": y_hat, "RMSE": rmse_model, "model": model}

    dct_med = {"prediction": np.repeat(pred_med, len(y_test)), "RMSE": rmse_med}

    dct_mean = {"prediction": np.repeat(pred_mean, len(y_test)), "RMSE": rmse_mean}

    with open("resources/models/baseline_model.pkl", "wb") as f:
        pickle.dump(dct_model, f)

    with open("resources/models/baseline_median.pkl", "wb") as f:
        pickle.dump(dct_med, f)

    with open("resources/models/baseline_mean.pkl", "wb") as f:
        pickle.dump(dct_mean, f)
