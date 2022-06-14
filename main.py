import argparse
from melkor.models import SKLearnModelPipeline
from melkor.datasets import AmesDataset
from melkor.utils import config_parser
from pathlib import Path
from sklearn import model_selection, metrics


def main(config: dict, data: str, model: str):

    ames_data = AmesDataset(Path(data), target_col="sale_price")

    X, y = ames_data.get_data()

    pipeline = SKLearnModelPipeline(
        config, cat_cols=ames_data.get_cat_cols(), num_cols=ames_data.get_num_cols()
    )

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        test_size=config["model_pipeline"]["test_size"],
        shuffle=config["model_pipeline"]["shuffle_train_split"],
    )

    pipeline.fit(X_train, y_train)

    pipeline.save_model_pipeline(model)

    y_hat = pipeline.predict(X_test)

    rmse = metrics.mean_squared_error(y_test, y_hat, squared=False)

    print(rmse)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Based On Sklearn Template")
    parser.add_argument(
        "-c",
        "--config",
        default="configs/config.yaml",
        type=str,
        help="config file path (default: configs/config.yaml)",
    )

    parser.add_argument(
        "-d",
        "--data",
        default="resources/data",
        type=Path,
        help="input data path (default: resources/data)",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="resources/models/latest.pkl",
        type=Path,
        help="output model path (default: resources/models/latest.pkl)",
    )

    args = parser.parse_args()

    config = config_parser(args.config)

    main(config, args.data, args.model)
