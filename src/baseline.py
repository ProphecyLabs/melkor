import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

rndseed = 42

if __name__ == "__main__":

    # import data
    df = pd.read_csv("resources/data/master.csv")

    # define vector of dependend variable
    y = df.sale_price

    # define matrix of features
    X = df[[i for i in df.columns if i != "sale_price"]]

    # split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=rndseed
    )

    # define model, uses default alpha value of 1
    model = Ridge(solver="cholesky")

    # fit model
    model.fit(X, y)

    # generate model predictions
    y_hat = model.predict(X_test)

    # calculate rmse for model predictions
    rmse_model = mean_squared_error(y_test, y_hat, squared=False)

    # calculate median of train set dependend variable
    pred_med = np.median(y_train)

    # calculate rmse for median prediction
    rmse_med = mean_squared_error(
        y_test, np.repeat(pred_med, len(y_test)), squared=False
    )

    # calculate mean of train set dependend variable
    pred_mean = np.mean(y_train)

    # calculate rmse for mean prediction
    rmse_mean = mean_squared_error(
        y_test, np.repeat(pred_mean, len(y_test)), squared=False
    )

    # generate output dictionary for model prediction
    dct_model = {"prediction": y_hat, "RMSE": rmse_model, "model": model}

    # generate output dictionary for median prediction
    dct_med = {"prediction": np.repeat(pred_med, len(y_test)), "RMSE": rmse_med}

    # generate output dictionary for mean prediction
    dct_mean = {"prediction": np.repeat(pred_mean, len(y_test)), "RMSE": rmse_mean}

    # save output files
    with open("resources/models/baseline_model.pkl", "wb") as f:
        pickle.dump(dct_model, f)

    with open("resources/models/baseline_median.pkl", "wb") as f:
        pickle.dump(dct_med, f)

    with open("resources/models/baseline_mean.pkl", "wb") as f:
        pickle.dump(dct_mean, f)
