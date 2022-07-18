import argparse
from melkor.models import SKLearnModelPipeline
from melkor.datasets import AmesDataset
from melkor.utils import config_parser, eval_regression
from pathlib import Path
from sklearn import model_selection
import pandas as pd
from typing import Union
import mlflow


def log_data(
    frame: Union[pd.DataFrame, pd.Series], output_dir: Path, name: str
) -> None:
    """MLFlow requires artifacts to be saved manually.
    Save pandas data to the output directory so that it can be logged by mlflow as an artifact.

    Parameters
    ----------
    frame: {pandas Series or DataFrame} the data that need to be saved.
    output_dir: {Path} the path of the outputs of the run
    name: {str} name of the artifact that's being saved i.e. training_data

    """
    data_dir = output_dir / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    data_path = data_dir / f"{name}.csv"
    frame.to_csv(data_path)
    mlflow.log_artifact(data_path, "data")

def main(config: dict, data_path: dict, output_dir: Path):

    ames = AmesDataset(data_path["filename"], Path(data_path["root"]), data_path["url"])

    X, y = ames.get_data()

    pipeline = SKLearnModelPipeline(
        config, cat_cols=ames.get_cat_cols(), num_cols=ames.get_num_cols()
    )

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        test_size=config["model_pipeline"]["test_size"],
        shuffle=config["model_pipeline"]["shuffle_train_split"],
    )

    with mlflow.start_run():

        mlflow.log_dict(config, "configs/config.yaml")
        mlflow.set_tag(
            "model", config["model_pipeline"]["model"]["name"].split(".")[-1]
        )

        mlflow.log_params(pipeline.get_model().get_params())

        pipeline.fit(X_train, y_train)

        y_hat_test = pipeline.predict(X_test)
        y_hat_train = pipeline.predict(X_train)

        train_metrics = eval_regression(y_train, y_hat_train)
        test_metrics = eval_regression(y_test, y_hat_test)

        print("METRIC"+" "*16+"TRAIN "+" "*16+"TEST")
        for metric in train_metrics.keys():
            train_metric = train_metrics[metric]
            test_metric = test_metrics[metric]
            print(f"{metric} {' '*(20-len(metric))} {train_metric} {' '*(20-len(str(train_metric)))} {test_metric:.4f}")
            mlflow.log_metric("train-" + metric, train_metric)
            mlflow.log_metric("test-" + metric, test_metric)

        mlflow.sklearn.log_model(pipeline.pipeline, "model")
        log_data(X_test, output_dir, "x_test")
        log_data(y_test, output_dir, "y_test")
        log_data(X_train, output_dir, "x_train")
        log_data(y_train, output_dir, "y_train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Based On Sklearn Template")
    parser.add_argument(
        "-c",
        "--paths_config",
        default="configs/paths.yaml",
        type=str,
        help="config with paths (default: configs/paths.yaml)",
    )
    parser.add_argument(
        "-uri",
        "--tracking_uri",
        default="sqlite:///melkor-experiments.db",
        type=str,
        help="Tracking uri for mlflow",
    )
    parser.add_argument(
        "-name",
        "--experiment_name",
        default="ames-regression-experiment",
        type=str,
        help="Tracking uri for mlflow",
    )

    args = parser.parse_args()

    paths = config_parser(args.paths_config)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    config = config_parser(paths["config"])

    main(config, paths["data"], Path(paths["model"]))
