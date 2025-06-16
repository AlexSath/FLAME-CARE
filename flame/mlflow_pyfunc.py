import os, json

import tensorflow as tf
from tensorflow.keras.models import load_model
from csbdeep.models import CARE, Config
from mlflow.pyfunc import PythonModel

class MLFLOW_CARE_Model(PythonModel):
    def __init__(self, config_json_path):
        PythonModel.__init__(self)
        self.config = json.load(open(config_json_path, 'r'))

    def predict(self, *args, **kwargs):
        care_model = self.get_CARE()
        result = care_model.keras_model.predict(*args, **kwargs)
        del care_model
        return result
    
    def get_CARE(self):
        # care_config = Config(**self.config['CARE_Model']['CSBDeep_Config'])
        care_model = CARE(
            None,
            os.path.join(self.config['CARE_Model']['run_dir'])
        )
        return care_model


def _load_pyfunc(data_path):
    config_json = os.path.join(data_path, "model_config.json")
    model = MLFLOW_CARE_Model(config_json_path=config_json)
    return model