import os

from csbdeep.models import CARE, Config

from mlflow.pyfunc import PythonModel

class MLFLOW_CARE_Model(PythonModel, CARE):
    def __init__(self, *args, **kwargs):
        PythonModel.__init__(self)
        CARE.__init__(self, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.keras_model.predict(*args, **kwargs)
    

def _load_pyfunc(data_path):
    config_json = os.path.join(data_path, "model_config.json")
    weights_path = os.path.join(data_path, "weights_best.h5")
    csbdeep_config_args = config_json['CARE_Model']['CSBDeep_Config']
    model_config = Config(**csbdeep_config_args)
    model = MLFLOW_CARE_Model(config=model_config, name=config_json['CARE_Model']['name'], basedir=data_path)
    model.keras_model.load_weights(weights_path)