from csbdeep.models import CARE

from mlflow.pyfunc import PythonModel

class MLFLOW_CARE_Model(PythonModel, CARE):
    def __init__(self, *args, **kwargs):
        PythonModel.__init__(self)
        CARE.__init__(self, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.keras_model.predict(*args, **kwargs)