from melkor.datasets import AmesDataset
from pathlib import Path


class TestAmesDataset:
    ames_df = AmesDataset(Path("resources/data"), target_col="sale_price")

    def test_data_shape(self):
        assert sum(self.ames_df.data.shape) > 0, "DataFrame is empty"

    def test_data_download(self):
        self.ames_df.download()
        assert (
            self.ames_df.root / self.ames_df.filename
        ).stat().st_size > 10**4, "Download size too small, likely corrupted file"

    def test_data_null(self):
        assert (
            self.ames_df.data.isnull().values.any() == 0
        ), "Cleaned dataset contains null values"
