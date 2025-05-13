class TileJSONNotFoundError(Exception):
    "Raise if the JSON corresponding to the FLAME Image cannot be found"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class FLAMEImageError(Exception):
    "Raise if the FLAME Image could not be initialized for any reason"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)