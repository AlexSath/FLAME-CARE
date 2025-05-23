import logging
import os
import gc
from typing import Any, Union

import numpy as np
from tifffile import TiffFile

from .tile import TileData
from .error import FLAMEImageError, TileDataError
from .utils import _validate_int_greater_than_zero
from .utils import _apply_bidirectional_correction

class FLAMEImage():
    def __init__(
            self, 
            impath: str, 
            jsonext: str, 
            checkChannels: bool = True,
            overrideNChannels: int = None,
            checkFrames: bool = True,
            overrideNFrames: int = None,
            checkZs: bool = False,
            requireBidirectionalCorrection: bool=False
        ) -> None:
        """
        FLAMEImage object.

        Parameters:
         - impath (str): the absoulte path to the provided image, expected to be in tif format.
         - jsonext (str): the expected string for the json paired with the image tif.
         - checkChannels (bool): whether to confirm the number of channels listed in the JSON matches the tif data (DEFAULT = True).
         - overrideNChannels (None, int): if None, don't override #channels. If not None, will override #channels with provided value.
         - checkFrames (bool): whether to confirm the number of frames listed in the JSON matches the tif data (DEFAULT = True).
         - overrideNFrames (None, int): if None, don't override #frames. If not None, will override #frames with provided value
         - checkZs (bool): whether to confirm the number of Zs listed in the JSON matches the tif data (DEFAULT = False).

        NOTE: overrideNZs is not required because tifffile package does not check for number of Zs when unpacking a tif of 
        size NYX to ZFCYX. This means that with NChannel and NFrame overrides being set, the number of Zs will be dynamically
        unpacked based on N (the number of pages in the raw tiff file).

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
            self.tileData = TileData(self.jsonpath, requireBidirectionalCorrection)
            self.imageData = None
            self.axes_shape = None
            self.imShape = None
            self.imDType = None
            self.isOpen = False
            self.checkChannels = checkChannels
            self.hasChannels = False
            self.checkFrames = checkFrames
            self.hasFrames = False
            self.checkZs = checkZs
            self.hasZs = False
            self.overrideNChannels = _validate_int_greater_than_zero(
                data=overrideNChannels, logger=self.logger, accept_nonetype=True, accept_float=False
            )
            self.overrideNFrames = _validate_int_greater_than_zero(
                data=overrideNFrames, logger=self.logger, accept_nonetype=True, accept_float=False
            )
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
                im = TiffFile(self.impath)
                assert im.is_scanimage, f"Only tiffs of type ScanImage are supported, not tiffs of type {im.flags}"

                # overriding scanimage metadata to force proper output shape for the tifffile.
                # so far, this has been seen when a FLAME Image is taken with N number of frames,
                # but only the frame aggregate has been saved (so the true frame should be 1 instead of N).
                if self.overrideNFrames is not None:
                    im.scanimage_metadata['FrameData']['SI.hStackManager.framesPerSlice'] = self.overrideNFrames
                    self.tileData.framesPerTile = self.overrideNFrames
                if self.overrideNChannels is not None:
                    im.scanimage_metadata['FrameData']['SI.hChannels.channelSave'] = self.overrideNChannels
                    self.tileData.channelsAcquired = list(range(self.overrideNChannels))
                if (
                    self.overrideNChannels is not None or 
                    self.overrideNFrames is not None
                ): im.series = im._series_scanimage() # this is required to force an update with overridden 
                return_image = im.asarray()

                # Apply the bidirectional scanning correction if wasfound in tileData.txt
                if 'bidirectionalCorrection' in self.tileData.availableData:
                    return_image = _apply_bidirectional_correction(return_image, self.tileData.bidirectionalCorrection)
                return return_image
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
            Zs = len(self.tileData.tileZs) if isinstance(self.tileData.tileZs, np.ndarray) else self.tileData.tileZs
            frames = self.tileData.framesPerTile
            channels = len(self.tileData.channelsAcquired)
            if self.checkChannels and self.checkFrames and self.checkZs: 
                assert this_dim == Zs * frames * channels
                self.hasChannels, self.hasFrames, self.hasZs = True, True, True
                self.axes_shape = "ZFCYX"
            elif self.checkChannels and self.checkFrames: # will be most common
                assert this_dim == channels * frames
                self.hasChannels, self.hasFrames = True, True
                self.axes_shape = "FCYX"
            elif self.checkZs and self.checkFrames: 
                assert this_dim == Zs * frames
                self.hasZs, self.hasFrames = True, True
                self.axes_shape = "ZFYX"
            elif self.checkChannels and self.checkZs:
                assert this_dim == channels * Zs
                self.hasChannels, self.hasZs = True, True
                self.axes_shape = "ZCYX"
            elif self.checkChannels:
                assert this_dim == channels
                self.hasChannels = True
                self.axes_shape = "CYX"
            elif self.checkFrames:
                assert this_dim == frames
                self.hasFrames = True
                self.axes_shape = "FYX"
            elif self.checkZs:
                assert this_dim == Zs
                self.hasZs = True
                self.axes_shape = "ZYX"
            else: # don't check anything; bad practice so raise exception
                raise Exception(f"No dim checks provided for tiff. Cannot verify completeness.")
        except Exception as e:
            self.logger.exception(f"Could not verify completeness of tiff from {self}.\n" \
                                  + f"Dim: {self.imShape} | Zs: {type(Zs)} | Frames: {frames} | Channels: {channels}" \
                                  + f"\nERROR: {e}")
            raise FLAMEImageError(f"Could not verify completeness of tiff from {self}.\n" \
                                  + f"Dim: {self.imShape} | Zs: {type(Zs)} | Frames: {frames} | Channels: {channels}" \
                                  + f"\nERROR: {e}")

    def get_frames(self, start: int, end: int, op: str="add") -> np.array:
        # assumes [Frame, Channels, Y, X] shape of tiff
        try:
            assert self.axes_shape == "FCYX", f"Axes should be of shape 'FCYX', not {self.axes_shape}"
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