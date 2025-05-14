import logging
import os
import gc

import numpy as np
import tifffile as tiff

from .tile import TileData
from .error import FLAMEImageError, TileDataError

class FLAMEImage():
    def __init__(self, impath: str, jsonext: str, 
                 checkChannels: bool = True,
                 checkFrames: bool = True,
                 checkZs: bool = False
                 ) -> None:
        """
        FLAMEImage object.

        Parameters:
         - impath (str): the absoulte path to the provided image, expected to be in tif format.
         - jsonext (str): the expected string for the json paired with the image tif.
         - checkChannels (bool): whether to confirm the number of channels listed in the JSON matches the tif data (DEFAULT = True).
         - checkFrames (bool): whether to confirm the number of frames listed in the JSON matches the tif data (DEFAULT = True).
         - checkZs (bool): whether to confirm the number of Zs listed in the JSON matches the tif data (DEFAULT = False).

        Attributes:
         - logger (Logger): the logger object with the name "main"
         - impath (str): the absolute path to the provided image
         - jsonpath (str): the path to the JSON corresponding to the tiff. Dynamically determined using params impath and jsonext.
         - tileData (TileData): the TileData object created from jsonpath
         - imageData (None or np.array): the placeholder variable for image data if opened
         - isOpen (bool): whether the tif data is loaded into the imageData attribute
         - checkChannels, checkFrames, checkZs (bool): see above. Used in checkForCompleteness() to validate the tif data.

        Methods:
         - get_json_path(ext: str) -> str: returns the json path given the impath and provided json extension
         - openImage() -> None: loads image data from impath into imageData attribute. Sets isOpen attribute to True.
         - closeImage() -> None: cleans image data from imageData attribute and system memory. Sets isOpen to False.
         - raw() -> np.array: returns raw image data from tif as numpy array
         - checkForCompleteNess() -> None: will throw FlameImageError if tif at provided path has shape inconsistent with tileData.
         - getFrames(start: int, end: int, op: str = "add") -> np.array: aggregate frames from [start:end] using operation 'op'.
        """
        self.logger = logging.getLogger("main")
        self.logger.info(f"Loading FLAME Image from {impath}")

        try:
            self.impath = impath
            self.jsonpath = self.get_json_path(jsonext)
            self.tileData = TileData(self.jsonpath)
            self.imageData = None
            self.imShape = None
            self.imDType = None
            self.isOpen = False
            self.checkChannels = checkChannels
            self.hasChannels = False
            self.checkFrames = checkFrames
            self.hasFrames = False
            self.checkZs = checkZs
            self.hasZs = False
        except Exception as e:
            self.logger.error(f"Could not initialize FLAMEImage object from {impath}")
            raise FLAMEImageError(f"Could not initialize FLAMEImage object from {impath}")
        
        self.checkForCompleteness()
        self.logger.info(f"Loaded FLAME Image tile data with {len(self.tileData)} data points")

    def get_json_path(self, ext: str) -> str:
        imname, imext = os.path.splitext(self.impath)
        jsonpath = f"{imname}.{ext}"
        if os.path.isfile(jsonpath):
            return jsonpath
        else:
            self.logger.error(f"Could not find JSON associated with the image {imname} ({ext} was provided as JSON extention)")
            raise TileDataError(f"Could not find JSON associated with the image {imname} ({ext} was provided as JSON extention)")

    def openImage(self) -> None:
        """Will open the image into the memory of the object."""
        self.imageData = self.raw()
        self.imShape = self.imageData.shape
        self.imDType = self.imageData.dtype
        self.isOpen = True

    def closeImage(self) -> None:
        del self.imageData
        gc.collect()
        self.imageData = None
        self.isOpen = False
    
    def raw(self) -> np.array:
        if self.isOpen:
            return self.imageData
        else:
            try:
                return tiff.imread(self.impath)
            except Exception as e:
                self.logger.error(f"Could not load tiff from {self}.\nERROR: {e}")
                raise FLAMEImageError(f"Could not load tiff from {self}.\nERROR: {e}")
        
    def checkForCompleteness(self) -> None:
        """
        Caution!! Assumes image will be in *XY format
        * is the wildcard dimension, which theoretically could contain intercolated Z, channel, and frame information     
        """
        if self.imShape is None:
            self.openImage() # if imshape is none, that means image has never been opened.
            self.closeImage() # by cycling imshape, self.imShape gets set.

        this_dim = np.cumprod(self.imShape[:-2])[-1] # take product of all channels before XY
        try:
            Zs = self.tileData.tileZs
            frames = self.tileData.framesPerTile
            channels = len(self.tileData.channelsAcquired)
            if self.checkChannels and self.checkFrames and self.checkZs: 
                assert this_dim == Zs * frames * channels
                self.hasChannels, self.hasFrames, self.hasZs = True, True, True
            elif self.checkChannels and self.checkFrames: # will be most common
                assert this_dim == channels * frames
                self.hasChannels, self.hasFrames = True, True
            elif self.checkZs and self.checkFrames: 
                assert this_dim == Zs * frames
                self.hasZs, self.hasFrames = True, True
            elif self.checkChannels and self.checkZs:
                assert this_dim == channels * Zs
                self.hasChannels, self.hasZs = True, True
            elif self.checkChannels:
                assert this_dim == channels
                self.hasChannels = True
            elif self.checkFrames:
                assert this_dim == frames
                self.hasFrames = True
            elif self.checkZs:
                assert this_dim == Zs
                self.hasZs = True
            else: # don't check anything; bad practice so raise exception
                raise Exception(f"No dim checks provided for tiff. Cannot verify completeness.")
        except Exception as e:
            self.logger.exception(f"Could not verify completeness of tiff from {self}.\n" \
                                  + f"Dim: {self.imShape} | Zs: {Zs} | Frames: {frames} | Channels: {channels}" \
                                  + f"\nERROR: {e}")
            raise FLAMEImageError(f"Could not verify completeness of tiff from {self}.\n" \
                                  + f"Dim: {self.imShape} | Zs: {Zs} | Frames: {frames} | Channels: {channels}" \
                                  + f"\nERROR: {e}")

    def get_frames(self, start: int, end: int, op: str="add") -> np.array:
        # assumes [Frame, Channels, X, Y] shape of tiff
        try:
            frames = self.raw()[start:end,...]
            
            if op == "add":
                frames = np.sum(frames, axis=0)
            else:
                self.logger.warning(f"Did not recognize operation {op} for frame aggregation. Performing 'addition' instead...")
                frames = np.sum(frames, axis=0)

            assert not np.all(frames == 0)

        except Exception as e:
            self.logger.exception(f"Failed to get frames from {self}.\nERROR: {e}")
            raise FLAMEImageError(f"Failed to get frames from {self}.\nERROR: {e}")

        return frames

    def __repr__(self) -> str:
        return f"FLAME Image @{hex(id(self))} from {self.impath}"

    def __str__(self) -> str:
        return f"FLAME Image @{hex(id(self))} from {self.impath}"