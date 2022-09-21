import argparse
from melkor.models import SKLearnModelPipeline
from melkor.datasets import AmesDataset
from pathlib import Path
import mlflow


def main(data_path: Path, model_uri: str):

    ad = AmesDataset(Path(data_path), target_col="sale_price")

    pipeline = SKLearnModelPipeline(inference=True)
    pipeline.load_model_from_uri(model_uri)

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
        "-t",
        "--tracking-uri",
        default="sqlite:///melkor-experiments.db",
        type=str,
        help="Tracking uri for mlflow",
    )

    parser.add_argument(
        "-m",
        "--model_uri",
        required=True,
        type=str,
        help="MLFLow model uri",
    )

    args = parser.parse_args()
    
    mlflow.set_tracking_uri(args.tracking_uri)

    main(args.data, args.model_uri)