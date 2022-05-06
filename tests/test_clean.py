import pandas as pd


class FeatureNotFoundError(Exception):
    """Raised when Features are not found in the dataset"""

    pass


if __name__ == "__main__":
    df = pd.read_csv("resources/df_clean.csv")

    check_feature = "new_feature"

    if check_feature not in df.columns:
        raise FeatureNotFoundError(f"{check_feature} not found in dataset")
