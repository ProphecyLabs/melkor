import argparse
from melkor.models import SKLearnModelPipeline
from melkor.datasets import AmesDataset
from melkor.utils import config_parser, eval_regression
from pathlib import Path
from sklearn import model_selection


def main(config: dict, data_path: dict, model_uri: str):

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

    pipeline.fit(X_train, y_train)

    pipeline.save_model_pipeline(model_path)

    y_hat_test = pipeline.predict(X_test)
    y_hat_train = pipeline.predict(X_train)

    train_metrics = eval_regression(y_train, y_hat_train)
    test_metrics = eval_regression(y_test, y_hat_test)

    print("METRIC"+" "*16+"TRAIN "+" "*16+"TEST")
    for metric in train_metrics.keys():
        train_metric = train_metrics[metric]
        test_metric = test_metrics[metric]
        print(f"{metric} {' '*(20-len(metric))} {train_metric} {' '*(20-len(str(train_metric)))} {test_metric:.4f}")


if __name__ == "__main__":

    paths = config_parser("configs/paths.yaml")

    config = config_parser(paths["config"])

    main(config, paths["data"], paths["model"])
