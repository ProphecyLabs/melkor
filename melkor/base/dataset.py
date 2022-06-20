import pandas as pd
import numpy as np
from pathlib import Path
import requests
from typing import Tuple, Callable


class PandasBaseDataset:
    """Base dataset class using pandas DataFrames

    Args:
        filename (str): filename for saved file
        root (str, optional): root path for file storage. Defaults to Path("./resources/data").
        URL (str, optional): URL of dataset. Defaults to None.
    """

    def __init__(
        self, filename: str, root: Path = Path("./resources/data"), URL: str = None
    ):

        self.root = root.absolute()
        self.URL = URL
        self.filename = filename
        self.filepath = self.root / self.filename

    def read(self, read_func: Callable, **kwargs):
        """Read in dataset from filepath

        Args:
            read_func (Callable): pandas import function
        """

        if ~self.root.exists():
            self.download()

        self.data = read_func(**kwargs)

    def download(self):
        """Download file"""
        response = requests.get(self.URL)

        assert response.ok, "Could not fetch dataset"

        with open(self.filepath, "wb") as f:
            f.write(response.content)

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into features and target

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: pandas DataFrames for features and target
        """
        return (self.data.drop(self.target_col, axis=1), self.data[[self.target_col]])

    def get_cat_cols(self) -> list:
        """get list of categorical columns

        Returns:
            list: list of categorical columns
        """

        return [
            i
            for i in self.data.select_dtypes(include=object).columns.tolist()
            if i != self.target_col
        ]

    def get_num_cols(self) -> list:
        """get list of numerical columns

        Returns:
            list: list of numerical colummns
        """
        return [
            i
            for i in self.data.select_dtypes(include=np.number).columns.tolist()
            if i != self.target_col
        ]
