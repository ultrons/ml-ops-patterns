from kfp import dsl
from kfp.v2 import compiler
from typing import NamedTuple
from kfp.v2 import dsl
from kfp.v2.dsl import (
        component,
        InputPath,
        OutputPath,
        Input,
        Output,
        Artifact,
        Dataset,
        Model,
        ClassificationMetrics,
        Metrics,

)

from kfp.v2.google.client import AIPlatformClient
import os

class complib:
    def __init__(self, container_uri, src_root):
        self.container_uri = container_uri
        self.src_root = src_root

    def fail_op(self):
        # Utility component fail with the given message
        @component(
                base_image=self.container_uri,
                # output_component_file=f'{self.src_root}/utils/fail_op.yaml',

        )
        def fail_op_def (message: str = "Metric is below threshhold"):
                raise ValueError(message)
        return fail_op_def

    def model_to_uri(self):
        # Return URI fromo the model object
        @component(
                base_image=self.container_uri,
               # output_component_file=f'{self.src_root}/utils/model_to_uri.yaml',

        )
        def model_to_uri_def (model: Input[Model] ) -> str:
                return model.uri
        return model_to_uri_def
