import os, json, logging, glob
from types import NoneType
from typing import Union, List, Tuple

from natsort import natsorted
import pandas as pd

from .error import FLAMEIOError

LOGGER = logging.getLogger("FLAMEIO")

def get_unshared_path(p1, p2):
    n_base = len(p1.split(os.path.sep))
    n_target = len(p2.split(os.path.sep))
    assert n_base < n_target, f"Path 1 {p1} must be smaller than Path 2 {p2}"
    return os.path.sep.join(p2.split(os.path.sep)[n_base:])


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
        this_id: str, 
        key: Union[str, List[str]]=['FLAME_Dataset', 'id'],
        file_ext: str="json",
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
    glob_address = os.path.join(input_direc, f"*.{file_ext}")
    hits = glob.glob(glob_address)
    assert len(hits) > 0, f"Not hits found using glob {glob_address}"

    matching_files = 0
    for hit in hits:
        try:
            try:
                json_dict = json.load(open(hit, 'r'))
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
                LOGGER.warning(f"Assess whether {this_id} was in JSON from path {hit}. 'fail_ok' set to True. Continuing...")
                continue
            else:
                raise FLAMEIOError(f"Assess whether {this_id} was in JSON from path {hit}. 'fail_ok' set to False. Exiting.\n{e.__class__.__name__}: {e}")
            
        if json_dict == this_id: matching_files += 1

    if matching_files == 0 and not fail_ok:
        LOGGER.exception(f"'fail_ok' set to False and found no JSONs found with {this_id} in {input_direc} kiven key {key}.")
        raise FLAMEIOError(f"'fail_ok' set to False and found no JSONs found with {this_id} in {input_direc} kiven key {key}.")
    elif matching_files == 0 and fail_ok:
        LOGGER.warning(f"'fail_ok' set to True and found no JSONs found with {this_id} in {input_direc} kiven key {key}. Returning None.")
        return None, None
    elif matching_files > 1:
        LOGGER.exception(f"Multiple JSONs found with {this_id} in {input_direc} kiven key {key}. Resolve conflict and try again.")
        raise FLAMEIOError(f"Multiple JSONs found with {this_id} in {input_direc} kiven key {key}. Resolve conflict and try again.")
    else:
        return output_hit, output_dict


def flame_paths_from_ids(
        root_dir: str, 
        index_path: str, 
        id_list: List[int], 
        pd_sep: str=",",
        id_col: str="id",
        path_col: str="image",
        tile_data_ext: str="tileData.txt",
        accept_missing: bool=False
    ) -> List[str]:
    """
    Description: Will look for flame images based on their unique IDs given a root directory and a
    table matching IDs to filepaths.

    Args:
     - root_dir (str): a path to the root directory when all images will be searched for
     - index_path (str): a path to the table (DataFrame-like) matching unique image ids to relative image paths
     - id_list (list[int]): single image id or list of image ids
     - pd_sep (str): the separation character that Pandas should use to parse 'index_path' into a DataFrame. DEFAULT=','
     - id_col (str): the name of the column storing image ids. DEFAULT='id'
     - path_col (str): the name of the column storing relative image paths. DEFAULT='image'
     - tile_data_ext (str): the extension used to search for every image's associated tileData JSON. DEFAULT='tileData.txt'
     - accept_missing (bool): whether to skip missing ids or raise an error when they are encountered. DEFAULT=False
    """

    assert os.path.isdir(root_dir), f"Could not find directory at {root_dir}"
    assert os.path.isfile(index_path), f"Provided index path is not a file ({index_path})"

    try:
        df = pd.read_csv(index_path, sep=pd_sep)
    except Exception as e:
        LOGGER.error(f"Could not open dataframe from provided 'index_path' ({index_path}).\n{e.__class__.__name__}: {e}")
        raise FLAMEIOError(f"Could not open dataframe from provided 'index_path' ({index_path}).\n{e.__class__.__name__}: {e}")
    
    id_dict = {}
    for this_id, path in zip(df[id_col], df[path_col]):
        id_dict[this_id] = path

    if not isinstance(id_list, list): id_list = [id_list]

    output_paths = []
    for this_id in id_list:
        try:
            this_rel_path = id_dict[this_id]
        except KeyError as e:
            LOGGER.warning(f"Could not find image of index {this_id} in {index_path}.")

        try:
            this_full_path = os.path.join(root_dir, this_rel_path)
            assert os.path.isfile(this_full_path), f"Could not file image at specified path {this_full_path}."
            this_tileData_path = f"{os.path.splitext(this_full_path)[0]}.{tile_data_ext}"
            assert os.path.isfile(this_full_path), f"Could not file associated tileData at {this_tileData_path}."
            output_paths.append(this_full_path)
        except Exception as e:
            if accept_missing:
                LOGGER.warning(
                    f"Could not verify the existance of image ({os.path.basename(this_full_path)})" \
                    + f"and/or tile data (.{tile_data_ext}) from {os.path.dirname(this_full_path)}. Skipping!" \
                    + f"\n{e.__class__.__name__}: {e}"
                )
                continue
            else:
                LOGGER.exception(
                    f"Could not verify the existance of image ({os.path.basename(this_full_path)})" \
                    + f"and/or tile data (.{tile_data_ext}) from {os.path.dirname(this_full_path)}." \
                    + f"\n{e.__class__.__name__}: {e}"
                )
                raise FLAMEIOError(
                    f"Could not verify the existance of image ({os.path.basename(this_full_path)})" \
                    + f"and/or tile data (.{tile_data_ext}) from {os.path.dirname(this_full_path)}." \
                    + f"\n{e.__class__.__name__}: {e}"
                )
    
    return output_paths

