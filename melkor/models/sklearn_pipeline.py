import sklearn
import pandas as pd
import numpy as np
import pickle
from nptyping import NDArray, Float32
from typing import Union
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import ensemble, preprocessing

# TODO: Add scoring metrics
# TODO: Change y to 1d array


class SKLearnModelPipeline:
    def __init__(
        self,
        config: dict = {},
        cat_cols: list = [],
        num_cols: list = [],
        inference: bool = False,
    ) -> None:
        self.config = config
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        if not inference:
            self.init_train()

    def init_train(self):
        self.config = self.config.get("model_pipeline")
        self.pipeline = self.create_pipeline()

    def create_pipeline(self):

        cat_transforms = self.config.get("data_transformation").get("categorical")
        num_transforms = self.config.get("data_transformation").get("numerical")

        categorical_transformer = self.parse_subpipeline(cat_transforms)
        numeric_transformer = self.parse_subpipeline(num_transforms)

        data_transformation = ColumnTransformer(
            transformers=[
                ("numerical", numeric_transformer, self.num_cols),
                ("categorical", categorical_transformer, self.cat_cols),
            ]
        )

        model_pipeline = self.parse_subpipeline({"model": self.config.get("model")})

        full_pipeline = Pipeline(
            [("data_transform", data_transformation), ("model", model_pipeline)]
        )

        return full_pipeline

    def save_model_pipeline(self, path) -> None:

        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load_model_pipeline(self, path) -> None:
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)

    def parse_subpipeline(self, subconfig) -> Pipeline:

        pipeline_steps = []
        for transformation_name, transformation_method in subconfig.items():
            class_hierarchy = transformation_method.get("name").split(".")
            sklearn_step = getattr(
                getattr(sklearn, class_hierarchy[0]), class_hierarchy[1]
            )(**transformation_method.get("params", {}))
            pipeline_steps.append((transformation_name, sklearn_step))

        return Pipeline(pipeline_steps)

    def fit(
        self, features: pd.DataFrame, target: Union[pd.DataFrame, pd.Series]
    ) -> None:
        self.pipeline.fit(features, target.values.ravel())

    def predict(self, features: pd.DataFrame):  # -> NDArray[Float32]
        return self.pipeline.predict(features)

    def get_model(self):
        return self.pipeline["model"].steps[0][1]

    def get_data_transformation(self):
        return self.pipeline["data_transform"].steps[0][1]
