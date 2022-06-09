from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple


class PandasBaseDataset(object):

    root = Path("./data").absolute()
    data = None
    target_col = None

    def read(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (self.data.drop(self.target_col, axis=1), self.data[[self.target_col]])

    def get_cat_cols(self) -> list:

        return [
            i
            for i in self.data.select_dtypes(include=object).columns.tolist()
            if i != self.target_col
        ]

    def get_num_cols(self) -> list:
        return [
            i
            for i in self.data.select_dtypes(include=np.number).columns.tolist()
            if i != self.target_col
        ]
