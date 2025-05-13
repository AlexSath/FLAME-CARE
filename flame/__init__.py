# TileData and FLAMEImage classes will live in namespace __all__
# This means they can be imported as: "from flame import TileData, FLAMEImage"
from .tile import TileData
from .image import FLAMEImage

__all__ = [k for k in globals().keys() if not k.startswith("_")]

# The custom errors will not live in namespace __all__
# This means that they can only be imported as:
#  - "from flame.tile import TileJSONNotFoundError"
#  - "from flame.image import FLAMEImageError"
from .tile import TileJSONNotFoundError
from .image import FLAMEImageError