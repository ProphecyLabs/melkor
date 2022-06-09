import argparse
from melkor.models import SKLearnModelPipeline
from melkor.datasets import AmesDataset
from pathlib import Path
from sklearn import *


def main(data_path: Path, model_path: Path):

    ad = AmesDataset(Path(data_path), target_col="sale_price")

    pipeline = SKLearnModelPipeline(inference=True)
    pipeline.load_model_pipeline(model_path)

    X, y = ad.get_data()

    y_hat = pipeline.predict(X)

    print(y_hat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Based On Sklearn Template")

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

    main(args.data, args.model)
