from sklearn.metrics import (
    mean_squared_error,
    explained_variance_score,
    max_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    hinge_loss,
    precision_score,
    recall_score,
    f1_score,
)
from typing import Union
import numpy as np
import pandas as pd
import yaml
from collections import defaultdict


def eval_regression(
    y_true: Union[list, np.array, pd.Series],
    y_predicted: Union[list, np.array, pd.Series],
) -> dict:
    """Evaluate a regression model and return a report on the most common metrics.

    Parameters
    ----------
    y_true: {list, numpy array, pandas Series} ground truth regression labels
    y_predicted: {list, numpy array, pandas Series} predicted regression labels

    Returns
    ----------
    metrics: {dict['metric name']: metric value} a dictionary of the most common regression metrics.

    """
    metrics = {}
    metrics["mse"] = mean_squared_error(y_true, y_predicted)
    metrics["rmse"] = mean_squared_error(y_true, y_predicted, squared=False)

    metrics["r2"] = r2_score(y_true, y_predicted)
    metrics["explained_variance"] = explained_variance_score(y_true, y_predicted)
    metrics["max_error"] = max_error(y_true, y_predicted)
    return metrics


def eval_classification(
    y_true: Union[list, np.array, pd.Series],
    y_predicted: Union[list, np.array, pd.Series],
    binary: bool = False,
) -> dict:
    """Evaluate a classification model and return a report on the most common metrics.

    Parameters
    ----------
    y_true: {list, numpy array, pandas Series} ground truth classification labels
    y_predicted: {list, numpy array, pandas Series} predicted classification labels

    Returns
    ----------
    metrics: {dict['metric name']: metric value} a dictionary of the most common classification metrics.

    """
    metrics = {}
    if binary:
        metrics["roc_auc"] = roc_auc_score(y_true, y_predicted)
        metrics["hinge"] = hinge_loss(y_true, y_predicted)

    metrics["accuracy"] = accuracy_score(y_true, y_predicted)

    metrics["precision_micro"] = precision_score(y_true, y_predicted, average="micro")
    metrics["precision_macro"] = precision_score(y_true, y_predicted, average="macro")

    metrics["recall_micro"] = recall_score(y_true, y_predicted, average="micro")
    metrics["recall_macro"] = recall_score(y_true, y_predicted, average="macro")

    metrics["f1_micro"] = f1_score(y_true, y_predicted, average="micro")
    metrics["f1_macro"] = f1_score(y_true, y_predicted, average="macro")

    return metrics


def to_snake(str_in, scream=False):
    """Convert string to snake_case or SNAKE_CASE"""
    # TODO: extend functionality to other characters that might need to be eliminated or replaced

    str_in = str_in.strip().replace(" ", "_").replace(",", "_").replace(":", "")

    if scream == False:
        return str_in.lower()
    else:
        return str_in.upper()


def config_parser(path: str):

    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return defaultdict(dict, data)
