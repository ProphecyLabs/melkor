import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
from melkor.base import PandasBaseDataset
from melkor.utils import to_snake


class AmesDataset(PandasBaseDataset):

    URL = "https://katrienantonio.github.io/hands-on-machine-learning-R-module-1/data/ames_python.csv"
    filename = "ames_data.csv"

    def __init__(self, root: Path, target_col: str) -> None:
        self.root = root.absolute()
        self.data = self.read()
        self.prepare_dataset()
        self.target_col = target_col

    def read(self) -> pd.DataFrame:
        if ~self.root.exists():
            self.download()

        self.data = pd.read_csv(self.root / self.filename, index_col=0)

        return self.data

    def download(self) -> None:
        response = requests.get(self.URL)

        assert response.ok, "Could not fetch dataset"

        with open(self.root / self.filename, "wb") as f:
            f.write(response.content)

    def prepare_dataset(self) -> None:

        self.data.columns = [to_snake(col) for col in self.data.columns]

        self.data = self.data.drop(["misc_val"], axis=1)

        # convert strings in categorical columns to SNAKE_CASE
        for col in self.data.select_dtypes(include=["object"]).columns:
            self.data[col] = self.data[col].apply(lambda x: to_snake(x, scream=True))

        # drop duplicate entries based on latitude and longitude columns
        self.data = self.data.sort_values(["year_sold", "mo_sold"]).drop_duplicates(
            subset=["latitude", "longitude"], keep="last", ignore_index=True
        )

        self.data[
            [
                "bsmt_full_bath",
                "bsmt_half_bath",
                "full_bath",
                "half_bath",
                "kitchen_abvgr",
                "fireplaces",
            ]
        ] = self.data[
            [
                "bsmt_full_bath",
                "bsmt_half_bath",
                "full_bath",
                "half_bath",
                "kitchen_abvgr",
                "fireplaces",
            ]
        ].astype(
            "int64"
        )

        self.data["building_age"] = datetime.now().year - self.data.year_built

        self.data["remod_age"] = self.data.year_remod_add - datetime.now().year

        # indicator whether building has been remodeled at all
        self.data["remod"] = (
            self.data["year_built"] == self.data["year_remod_add"]
        ).astype(int)

        # indicator whether house has a second floor
        self.data["second_flr"] = (self.data["second_flr_sf"] > 0).astype("int")

        # indicator whether house has masonry veneer
        self.data["second_flr"] = (self.data["mas_vnr_area"] > 0).astype("int")
