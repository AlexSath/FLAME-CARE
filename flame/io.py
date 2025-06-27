import os, json, logging, glob
from types import NoneType
from typing import Union, List, Tuple

from natsort import natsorted

from .error import FLAMEIOError

LOGGER = logging.getLogger("FLAMEIO")

def get_input_and_GT_paths(input_direc: str, input_frames: int, gt_frames: int) -> Tuple[List]:
    try:   
        low_paths = []
        GT_paths = []
        for root, dirs, files in os.walk(input_direc):
            for f in files:
                if f"frames{input_frames}" in f:
                    low_paths.append(os.path.join(root, f))
                elif f"frames{gt_frames}" in f:
                    GT_paths.append(os.path.join(root, f))
        assert len(low_paths) != 0 and len(GT_paths) != 0, f"Need to find more than 0 input image paths and GT image paths."
        assert len(low_paths) == len(GT_paths), f"Number of input image paths and GT image paths should be the same."
        LOGGER.info(f"Found {len(low_paths)} images in {input_direc}")
    except AssertionError as e:
        LOGGER.error(f"Incorrect image count in {input_direc}")
        raise FLAMEIOError(f"Incorrect image count in {input_direc}")
    except Exception as e:
        LOGGER.error(f"Unknown error occurred when finding images in input direc {input_direc}.\nERROR: {e}")
        raise FLAMEIOError(f"Unknown error occurred when finding images in input direc {input_direc}.\nERROR: {e}")

    low_paths = natsorted(low_paths)
    GT_paths = natsorted(GT_paths)

    return low_paths, GT_paths


def find_dataset_config(
        input_direc: str, 
        id: str, 
        key: Union[str, List[str]]=['FLAME_Dataset', 'id'],
        file_ext: str=".json",
        fail_ok: bool=False,
    ) -> Union[NoneType, Tuple[str, dict]]:
    """
    Searches through all files with extension 'file_ext' in provided 'input_direc' for a value of 'id'.
    Assumes files will be in json format, and will be opened using the builtin json I/O. Will use 'key'
    to search the dict. If key is a list of strings, each successive string will identify the nested
    sub-dictionary for search. Outcomes:
        1. If a match is found, return the entire dict stored by the associated .json.
        2. If no match is found, return None if 'fail_ok' is True. Otherwise throw FLAMEIOError.
        3. If multiple matches are found, throw FLAMEIOError

    Returns:
     - None IF no matches found and 'fail_ok' is True
     - tuple(output_hit (str), output_dict (dict)) IF a single match is found. String is path to json and dict is associated dictionary.
    """
    hits = glob.glob(os.path.join(input_direc, f"*.{file_ext}"))

    matching_files = 0
    for hit in hits:
        try:
            try:
                json_dict = json.read(open(hit, 'r'))
                output_dict = json_dict.copy()
                output_hit = hit
            except Exception as e:
                LOGGER.error(f"Failed to load json from {hit}.\n{e.__class__.__name__}: {e}")
                raise e
            
            if not isinstance(key, list): key = [key]
            for idx, k in enumerate(key):
                try:
                    json_dict = json_dict[k]
                except Exception as e:
                    LOGGER.error(f"Key {k} at subdictionary {idx+1} does not exist.\n{e.__class__.__name__}: {e}")
                    raise e
        except Exception as e:
            if fail_ok: 
                LOGGER.warning(f"Assess whether {id} was in JSON from path {hit}. 'fail_ok' set to True. Continuing...")
                continue
            else:
                raise FLAMEIOError(f"Assess whether {id} was in JSON from path {hit}. 'fail_ok' set to False. Exiting.\n{e.__class__.__name__}: {e}")
            
        if json_dict == id: matching_files += 1

    if matching_files == 0 and not fail_ok:
        LOGGER.exception(f"'fail_ok' set to False and found no JSONs found with {id} in {input_direc} kiven key {key}.")
        raise FLAMEIOError(f"'fail_ok' set to False and found no JSONs found with {id} in {input_direc} kiven key {key}.")
    elif matching_files == 0 and fail_ok:
        LOGGER.warning(f"'fail_ok' set to True and found no JSONs found with {id} in {input_direc} kiven key {key}. Returning None.")
        return None, None
    elif matching_files > 1:
        LOGGER.exception(f"Multiple JSONs found with {id} in {input_direc} kiven key {key}. Resolve conflict and try again.")
        raise FLAMEIOError(f"Multiple JSONs found with {id} in {input_direc} kiven key {key}. Resolve conflict and try again.")
    else:
        return output_hit, output_dict

    