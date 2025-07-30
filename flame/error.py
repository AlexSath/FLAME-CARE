class TileDataError(BaseException):
    "Raise if the JSON corresponding to the FLAME Image cannot be found"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class FLAMEImageError(BaseException):
    "Raise if the FLAME Image could not be initialized for any reason"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class FLAMEDtypeError(BaseException):
    "Raise if some dtype could not be appropriate verified"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class FLAMEEvalError(BaseException):
    "Raise if some error occurs during FLAME model evaluation"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class CAREDatasetError(BaseException):
    "Raise if some error occurs from a CARE dataset"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class CAREInferenceError(BaseException):
    "Raise if some error occurs during CARE inference"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class FLAMEIOError(BaseException):
    "Raise if some error occurs with input/output during FLAME/CARE"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class FLAMEMLFlowError(BaseException):
    "Raise if some error occurs during interfact with MLFlow in any FLAME dependency"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class FLAMECmdError(BaseException):
    "Raise if some error occurs when running command-line FLAME scripts"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class FLAMEPyMatlabError(BaseException):
    "Raise if some error occurs when Python tries to interface with MATLAB"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)