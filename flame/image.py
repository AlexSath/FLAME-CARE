import logging
import os

import numpy as np
import tifffile as tiff

from .tile import TileData, TileJSONNotFoundError

class FLAMEImage():
    def __init__(self, impath: str, jsonext: str) -> None:
        self.logger = logging.getLogger("main")

        try:
            self.impath = impath
            self.logger.info(f"Loading FLAME Image from {impath}")
            self.jsonpath = self.get_json_path(jsonext)
            self.tileData = TileData(self.jsonpath)
        except Exception as e:
            self.logger.exception(f"Could not initialize FLAMEImage object from {impath}")
            raise FLAMEImageError
        
        self.logger.info(f"Loaded FLAME Image tile data with {len(self.tileData)} data points")

    def get_json_path(self, ext: str) -> str:
        imname, imext = os.path.splitext(self.impath)
        jsonpath = f"{imname}.{ext}"
        if os.path.isfile(jsonpath):
            return jsonpath
        else:
            self.logger.exception(f"Could not find JSON associated with the image {imname} ({ext} was provided as JSON extention)")
            raise TileJSONNotFoundError

    def raw(self) -> np.array:
        try:
            return tiff.imread(self.impath)
        except Exception as e:
            self.logger.exception(f"Could not load tiff from {self.impath}.\nERROR: {e}")
            raise FLAMEImageError

    def get_frames(self, start: int, end: int, op: str="add") -> np.array:
        # assumes [Frame, Channels, X, Y] shape of tiff
        frames = self.raw()[start:end,...]
        if op == "add":
            frames = np.sum(frames, axis=0)
        else:
            self.logger.warning(f"Did not recognize operation {op} for frame aggregation. Performing 'addition' instead...")
            frames = np.sum(frames, axis=0)
        return frames

    def __repr__(self) -> str:
        return f"FLAME Image @ {self.impath}"

    def __str__(self) -> str:
        return f"FLAME Image @ {self.impath}"


class FLAMEImageError(Exception):
    "Raise if the FLAME Image could not be initialized for any reason"
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)