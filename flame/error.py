class TileDataError(Exception):
    "Raise if the JSON corresponding to the FLAME Image cannot be found"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class FLAMEImageError(Exception):
    "Raise if the FLAME Image could not be initialized for any reason"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class FLAMEDtypeError(Exception):
    "Raise if some dtype could not be appropriate verified"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class CAREInferenceError(Exception):
    "Raise if some error occurs during FLAME inference"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)