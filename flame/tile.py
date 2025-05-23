import logging
import json
import numpy as np

from .error import TileDataError
from .utils import _int_or_int_array, _float_or_float_array

class TileData():
    def __init__(
            self, 
            path: str,
            requireBidirectionalCorrection: bool=False,
        ) -> None:
        self.logger = logging.getLogger("main")
        self.availableData = []

        try: # loading tile data from JSON before parsing
            data = json.load(open(path, 'r'))
        except Exception as e:
            self.logger.error(f"Could not open JSON at provided path {path}.\nERROR:{e}")
            raise TileDataError(f"Could not open JSON at provided path {path}.\nERROR:{e}")

        try: # put all required keys in a try/except
            # Check if overarching TileData item is found in the JSON
            try:
                tileData = data['TileData']
            except KeyError as e:
                self.logger.exception("Required key 'TileData' not found in provided JSON")
                raise TileDataError("Required key 'TileData' not found in provided JSON")
            self.logger.info('Found TileData within provided JSON')

            # Within TileData, check for known keys and handle any missing keys.
            try:
                self.tileID = tileData.pop('tileID', None)
                self.availableData.append('tileID')
            except KeyError as e:
                self.logger.error("Required key 'tileID' not found in tile data JSON")
                raise TileDataError("Required key 'tileID' not found in tile data JSON")

            try:
                self.tileResolution = np.array(tileData.pop('tileResolution', None), dtype=np.uint16)
                self.availableData.append('tileResolution')
            except KeyError as e:
                self.logger.error("Required key 'tileResolution' not found in tile data JSON")
                raise TileDataError("Required key 'tileResolution' not found in tile data JSON")

            try:
                self.tileChannelsAvailable = np.array(tileData.pop('tileChannelsAvailable', None), dtype=np.uint8)
                self.availableData.append('tileChannelsAvailable')
            except KeyError as e:
                self.logger.error("Required key 'tileChannelsAvailable' not found in tile data JSON")
                raise TileDataError("Required key 'tileChannelsAvailable' not found in tile data JSON")

            try:
                self.channelsAcquired = np.array(tileData.pop('channelsAcquired', None), dtype=np.uint8)
                self.availableData.append('channelsAcquired')
            except KeyError as e:
                self.logger.error("Required key 'channelsAcquired' not found in tile data JSON")
                raise TileDataError("Required key 'channelsAcquired' not found in tile data JSON")
            
            try:
                self.channelsSaved = np.array(tileData.pop('channelsSaved', None), dtype=np.uint8)
                self.availableData.append('channelsSaved')
            except KeyError as e:
                self.logger.error("Required key 'channelsSaved' not found in tile data JSON")
                raise TileDataError("Required key 'channelsSaved' not found in tile data JSON")

            try:
                self.framesPerTile = int(tileData.pop('framesPerTile', None))
                self.availableData.append('framesPerTile')
            except KeyError as e:
                self.logger.error("Required key 'framesPerTile' not found in tile data JSON")
                raise TileDataError("Required key 'framesPerTile' not found in tile data JSON")

            try:
                self.tileScannedThisFile = _int_or_int_array(tileData.pop('tileScannedThisFile'), logger=self.logger)
                self.availableData.append('tileScannedThisFile')
            except KeyError as e:
                self.logger.error("Required key 'tileScannedThisFile' not found in tile data JSON")
                raise TileDataError("Required key 'tileScannedThisFile' not found in tile data JSON")

            try:
                self.tileZs = _int_or_int_array(tileData.pop('tileZs', None), logger=self.logger)
                self.availableData.append('tileZs')
            except KeyError as e:
                self.logger.error("Required key 'tileZs' not found in tile data JSON")
                raise TileDataError("Required key 'tileZs' not found in tile data JSON")
        
        except Exception as e: # for all non-key error exceptions
            self.logger.exception(f"Failed to load required information from tile data JSON.\nEXCEPTION: {e}")
            raise TileDataError(f"Failed to load required information from tile data JSON.\nEXCEPTION: {e}")

        self.logger.info("Successfully loaded all required keys from tile data JSON")

        # bidirectional correction is optional, but can be required if parameter set to True during TileData obj init.
        try:
            self.bidirectionalCorrection = int(tileData.pop('bidirectionalCorrection', None))
            self.availableData.append('bidirectionalCorrection')
        except Exception as e:
            if requireBidirectionalCorrection:
                self.logger.exception(f"Failed to load bidirectional correction when it was required.\nEXCEPTION: {e}")
                raise TileDataError(f"Failed to load bidirectional correction when it was required.\nEXCEPTION: {e}")
            self.logger.warning(f"'bidirectionalCorrection' could not be loaded from tile data JSON.\nERROR: {e}")
            self.bidirectionalCorrection = None

        # all other known tiledata stuff is not required, so can handle any exceptions with a warning        
        try:
            self.tileAffine = np.array(tileData.pop('tileAffine', None), dtype=np.float32)
            self.availableData.append('tileAffine')
        except Exception as e:
            self.logger.warning("'tileAffine' could not be loaded from tile data JSON")
            self.tileAffine = None

        try:
            self.displayAvgFactor = int(tileData.pop('displayAvgFactor', None))
            self.availableData.append('displayAvgFactor')
        except Exception as e:
            self.logger.warning("'displayAvgFactor' could not be loaded from tile data JSON")
            self.displayAvgFactor = None

        try:
            self.tileSamplePointXY = np.array(tileData.pop('tileSamplePointXY', None), dtype=np.float32)
            self.availableData.append('tileSamplePointXY')
        except Exception as e:
            # sample point XY is necessary for recognizing how many tiles were actually recorded
            self.logger.warning("'tileSamplePointXY' could not be loaded from tile data JSON")
            self.tileSamplePointXY = None
        
        try:
            self.tileSizeUm = np.array(tileData.pop('tileSizeUm', None), dtype=np.uint16)
            self.availableData.append('tileSizeUm')
        except Exception as e:
            self.logger.warning("'tileSizeUm' could not be loaded from tile data JSON")
            self.tileSizeUm = None

        try:
            self.tileCornerPtsUm = np.array(tileData.pop('tileCornerPtsUm', None), dtype=np.float32)
            self.availableData.append('tileCornerPtsUm')
        except Exception as e:
            self.logger.warning("'tileCornerPtsUm' could not be loaded from tile data JSON")
            self.tileCornerPtsUm = None

        try:
            self.isFastZ = bool(tileData.pop('isFastZ', None))
            self.availableData.append('isFastZ')
        except Exception as e:
            self.logger.warning("'isFastZ' could not be loaded from tile data JSON")
            self.isFastZ = None

        try:
            self.numTilesToScan = int(tileData.pop('numTilesToScan', None))
            self.availableData.append('numTilesToScan')
        except Exception as e:
            self.logger.warning("'numTilesToScan' could not be loaded from tile data JSON")
            self.numTilesToScan = None

        # If there are remaining keys that were not handled, let the user know.
        if len(tileData) != 0:
            self.logger.warning(f"Found {len(tileData)} unrecognized keys in tile data JSON. Continuing...")

    def __repr__(self):
        return f"TileData object with {len(self.availableData)} available datapoints. Use 'availableData' attribute to see them."
    
    def __str__(self):
        return f"TileData object with {len(self.availableData)} available datapoints. Use 'availableData' attribute to see them."
    
    def __len__(self):
        return len(self.availableData)