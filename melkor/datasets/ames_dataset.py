import pandas as pd
from datetime import datetime
from pathlib import Path
from melkor.base import PandasBaseDataset
from melkor.utils import to_snake


class AmesDataset(PandasBaseDataset):
    """Class for ames dataset

    Args:
    root (Path, optional): path for data storage. Defaults to Path("./resources/data").
    """

    URL = "https://katrienantonio.github.io/hands-on-machine-learning-R-module-1/data/ames_python.csv"
    filename = "ames_data.csv"
    target_col = "sale_price"

    def __init__(self, root: Path = Path("./resources/data")):
        super().__init__(self.filename, root, self.URL)
        self.read(pd.read_csv, filepath_or_buffer=self.filepath, index_col=0)
        self.prepare_dataset()

    def prepare_dataset(self):

        self.data.columns = [to_snake(col) for col in self.data.columns]

        self.data = self.data.drop(["misc_val"], axis=1)

        # convert strings in categorical columns to SNAKE_CASE
        for col in self.get_cat_cols():
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
