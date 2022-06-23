from melkor.datasets import AmesDataset
from melkor.utils import config_parser
from pathlib import Path

class TestAmesDataset:
    paths = config_parser("configs/paths.yaml")["data"]
    ames = AmesDataset(paths["filename"], Path(paths["root"]), paths["url"])

    def test_data_shape(self):
        assert sum(self.ames.data.shape) > 0, "DataFrame is empty"

    def test_data_download(self):
        self.ames.download()
        assert (
            self.ames.root / self.ames.filename
        ).stat().st_size > 10**4, "Download size too small, likely corrupted file"

    def test_data_null(self):
        assert (
            self.ames.data.isnull().values.any() == 0
        ), "Cleaned dataset contains null values"
