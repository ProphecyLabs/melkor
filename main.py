import argparse
from melkor.models import SKLearnModelPipeline
from melkor.datasets import AmesDataset
from melkor.utils import config_parser
from pathlib import Path
from sklearn import model_selection, metrics


def main(config: dict, data_path: dict, model_path: str):

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

    y_hat = pipeline.predict(X_test)

    rmse = metrics.mean_squared_error(y_test, y_hat, squared=False)

    print(rmse)


if __name__ == "__main__":

    paths = config_parser("configs/paths.yaml")

    config = config_parser(paths["config"])

    main(config, paths["data"], paths["model"])
