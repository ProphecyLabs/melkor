import pytest
from melkor.datasets import AmesDataset
from melkor.models import SKLearnModelPipeline
from melkor.utils import config_parser
from copy import copy
from sklearn import preprocessing, ensemble, model_selection
import os


class TestSKLearnModelPipeline:
    ames = AmesDataset()

    config = config_parser("configs/config.yaml")

    X, y = ames.get_data()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        test_size=config["model_pipeline"]["test_size"],
        shuffle=config["model_pipeline"]["shuffle_train_split"],
    )

    pipeline = SKLearnModelPipeline(
        config,
        cat_cols=ames.get_cat_cols(),
        num_cols=ames.get_num_cols(),
    )

    def test_pipeline_steps(self):
        assert (
            len(self.pipeline.pipeline.steps) > 0
        ), "Pipeline steps should be greater than 0"

    @pytest.mark.parametrize("feature_type", ["categorical", "numerical"])
    def test_create_pipeline_without_transformation(self, feature_type: str):
        new_config = copy(self.config)
        new_config["model_pipeline"]["data_transformation"][feature_type] = {}
        pipeline = SKLearnModelPipeline(
            new_config,
            cat_cols=self.ames.get_cat_cols(),
            num_cols=self.ames.get_num_cols(),
        )
        steps = pipeline.pipeline.named_steps.keys()
        assert "data_transform" in steps, "No data transformation in pipeline steps"
        data_dict = {
            i[0]: i[1] for i in pipeline.pipeline["data_transform"].transformers
        }
        assert (
            len(data_dict[feature_type].steps) == 0
        ), "Left out feature should contain no transformation steps"

    @pytest.mark.parametrize(
        "process_step",
        [
            ("numerical", "preprocessing.StandardScaler"),
            ("categorical", "preprocessing.OneHotEncoder"),
        ],
    )
    def test_data_transformation_steps(self, process_step: tuple):
        new_config = copy(self.config)
        new_config["model_pipeline"]["data_transformation"][process_step[0]] = {
            "foo": {"name": process_step[1]}
        }
        pipeline = SKLearnModelPipeline(
            new_config,
            cat_cols=self.ames.get_cat_cols(),
            num_cols=self.ames.get_num_cols(),
        )
        data_dict = {
            i[0]: i[1] for i in pipeline.pipeline["data_transform"].transformers
        }
        transformation_class = getattr(preprocessing, process_step[1].split(".")[1])
        assert isinstance(
            data_dict[process_step[0]].steps[0][1], transformation_class
        ), "Transformation class is not correct"

    def test_model_steps(self):
        model_class = getattr(
            ensemble, self.config["model_pipeline"]["model"]["name"].split(".")[1]
        )
        pipeline_model = self.pipeline.get_model()
        assert isinstance(pipeline_model, model_class), "Incorrect model class"

    def test_pipeline_save(self):
        pipeline_path = "resources/models/test.pkl"

        self.pipeline.fit(self.X_train, self.y_train)

        pipeline_copy = copy(self.pipeline)

        self.pipeline.save_model_pipeline(pipeline_path)

        self.pipeline.load_model_pipeline(pipeline_path)

        os.remove(pipeline_path)

        model_original = self.pipeline.get_model()
        model_copy = pipeline_copy.get_model()

        assert (
            model_original.get_params() == model_copy.get_params()
        ), "Saved and loaded pipeline should have matching parameters"

        assert (
            model_original.n_features_in_ == model_copy.n_features_in_
        ), "Saved and loaded pipeline should use same number of features"

        assert (
            model_original.feature_importances_ == model_copy.feature_importances_
        ).all(), "Saved and loaded pipeline should have the same feature importances"

        assert (
            self.pipeline.pipeline.feature_names_in_
            == pipeline_copy.pipeline.feature_names_in_
        ).all(), "Saved and loaded pipeline should use the same features"

    def test_model_predict(self):

        self.pipeline.fit(self.X_train, self.y_train)

        y_hat = self.pipeline.predict(self.X_test)

        assert sum(y_hat.shape) > 0, "Pipeline predictions should not be empty"
