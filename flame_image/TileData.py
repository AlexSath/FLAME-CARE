import logging
import numpy as np

class TileData():
    def __init__(self, data: dict) -> None:
        logger = logging.getLogger("main")
        self.availableData = []

        try: # put all required keys in a try/except
            # Check if overarching TileData item is found in the JSON
            try:
                tileData = data['TileData']
            except KeyError as e:
                logger.exception("'TileData' key not found in provided JSON")
                raise
            logger.info('Found TileData within provided JSON')

            # Within TileData, check for known keys and handle any missing keys.
            try:
                self.tileID = tileData.pop('tileID', None)
                self.availableData.append('tileID')
            except KeyError as e:
                logger.exception("'tileID' not found in tile data JSON")
                raise

            try:
                self.tileResolution = np.array(tileData.pop('tileResolution', None), dtype=np.uint16)
                self.availableData.append('tileResolution')
            except KeyError as e:
                logger.exception("'tileResolution' not found in tile data JSON")
                raise

            try:
                self.tileChannelsAvailable = np.array(tileData.pop('tileChannelsAvailable', None), dtype=np.uint8)
                self.availableData.append('tileChannelsAvailable')
            except KeyError as e:
                logger.exception("'tileChannelsAvailable' not found in tile data JSON")
                raise

            try:
                self.channelsAcquired = np.array(tileData.pop('channelsAcquired', None), dtype=np.uint8)
                self.availableData.append('channelsAcquired')
            except KeyError as e:
                logger.exception("'channelsAcquired' not found in tile data JSON")
                raise
            
            try:
                self.channelsSaved = np.array(tileData.pop('channelsSaved', None), dtype=np.uint8)
                self.availableData.append('channelsSaved')
            except KeyError as e:
                logger.exception("'channelsSaved' not found in tile data JSON")
                raise

            try:
                self.framesPerTile = int(tileData.pop('framesPerTile', None))
                self.availableData.append('framesPerTile')
            except KeyError as e:
                logger.exception("'framesPerTile' not found in tile data JSON")
                raise

            try:
                self.tileScannedThisFile = int(tileData.pop('tileScannedThisFile'))
                self.availableData.append('tileScannedThisFile')
            except KeyError as e:
                logger.exception("'tileScannedThisFile' not found in tile data JSON")
                raise

            try:
                self.tileZs = int(tileData.pop('tileZs', None))
                self.availableData.append('tileZs')
            except KeyError as e:
                logger.exception("'tileZs' not found in tile data JSON")
                raise
        
        except Exception as e: # for all non-key error exceptions
            logger.exception(f"Failed to load required information from tile data JSON.\nEXCEPTION: {e}")
            raise

        logger.info("Successfully loaded all required keys from tile data JSON")

        # all other known tiledata stuff is not required, so can handle any exceptions with a warning
        try:
            self.tileAffine = np.array(tileData.pop('tileAffine', None), dtype=np.float32)
            self.availableData.append('tileAffine')
        except Exception as e:
            logger.warning("'tileAffine' could not be loaded from tile data JSON")
            self.tileAffine = None

        try:
            self.displayAvgFactor = int(tileData.pop('displayAvgFactor', None))
            self.availableData.append('displayAvgFactor')
        except Exception as e:
            logger.warning("'displayAvgFactor' could not be loaded from tile data JSON")
            self.displayAvgFactor = None

        try:
            self.tileSamplePointXY = np.array(tileData.pop('tileSamplePointXY', None), dtype=np.float32)
            self.availableData.append('tileSamplePointXY')
        except Exception as e:
            # sample point XY is necessary for recognizing how many tiles were actually recorded
            logger.warning("'tileSamplePointXY' could not be loaded from tile data JSON")
            self.tileSamplePointXY = None
        
        try:
            self.tileSizeUm = np.array(tileData.pop('tileSizeUm', None), dtype=np.uint16)
            self.availableData.append('tileSizeUm')
        except Exception as e:
            logger.warning("'tileSizeUm' could not be loaded from tile data JSON")
            self.tileSizeUm = None

        try:
            self.tileCornerPtsUm = np.array(tileData.pop('tileCornerPtsUm', None), dtype=np.float32)
            self.availableData.append('tileCornerPtsUm')
        except Exception as e:
            logger.warning("'tileCornerPtsUm' could not be loaded from tile data JSON")
            self.tileCornerPtsUm = None

        try:
            self.isFastZ = bool(tileData.pop('isFastZ', None))
            self.availableData.append('isFastZ')
        except Exception as e:
            logger.warning("'isFastZ' could not be loaded from tile data JSON")
            self.isFastZ = None

        try:
            self.numTilesToScan = int(tileData.pop('numTilesToScan', None))
            self.availableData.append('numTilesToScan')
        except Exception as e:
            logger.warning("'numTilesToScan' could not be loaded from tile data JSON")
            self.numTilesToScan = None

        # If there are remaining keys that were not handled, let the user know.
        if len(tileData) != 0:
            logger.warning(f"Found {len(tileData)} unrecognized keys in tile data JSON. Continuing...")

    def __repr__(self):
        return f"TileData object with {len(self.availableData)} available datapoints. Use 'availableData' attribute to see them."
    
    def __str__(self):
        return f"TileData object with {len(self.availableData)} available datapoints. Use 'availableData' attribute to see them."
