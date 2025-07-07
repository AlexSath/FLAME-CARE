import os, logging, subprocess
from logging import Logger
from typing import Union, Any
from types import NoneType

import numpy as np
from numpy.typing import NDArray
from natsort import natsorted
from matlab import engine as matlab_engine

from .error import FLAMEDtypeError, CAREDatasetError, FLAMEMLFlowError

LOGGER = logging.getLogger("UTIL")

def _int_or_int_array(
        data: Any, 
        logger: Union[Logger, NoneType]=None,
        dtype: Union[int, np.integer]=int,
        accept_nonetype: bool=False,
    ) -> Union[NDArray[np.integer], np.integer, int]:
    """
    If data can be turned into an integer array, it will.
    If the data can't but matches dtype int, it will return a single integer.
    """
    if data is None and accept_nonetype:
        return data
    
    try:
        proc_data = dtype(data)
    except TypeError as e:
        try:
            proc_data = np.array(data, dtype=dtype)
        except ValueError as e:
            if logger is not None:
                logger.exception(f"Could not convert {data} to int or int array.\nERROR: {e}")
            raise FLAMEDtypeError(f"Could not convert {data} to int or int array.\nERROR: {e}")
    return proc_data


def _validate_int(
        data: Any, 
        logger: Union[Logger, NoneType]=None,
        dtype: Union[int, np.integer]=int,
        accept_nonetype: bool=False,
        accept_float: bool=False,
    ) -> Union[np.integer, int]:
    """
    Will ensure input data is an array.
    Will return None if input data is none and accept_nonetype is True.
    """
    if data is None and accept_nonetype:
        return data
    
    try:
        proc_data = dtype(data)
        if not accept_float:
            assert not np.issubdtype(type(data), np.floating), f"Data {data} is float, but accept_float is False"
    except TypeError as e:
        if logger is not None:
            logger.exception(f"Could not convert {data} to integer.\nERROR: {e}")
        raise FLAMEDtypeError(f"Could not convert {data} to integer.\nERROR: {e}")
    return proc_data


def _validate_int_greater_than_zero(
        data: Any, 
        logger: Union[Logger, NoneType]=None,
        dtype: Union[int, np.integer]=int,
        accept_nonetype: bool=False,
        accept_float: bool=False,
    ) -> Union[np.integer, int]:
    try:
        data = _validate_int(
            data=data, 
            logger=logger, 
            dtype=dtype, 
            accept_nonetype=accept_nonetype,
            accept_float = accept_float
        )
    except FLAMEDtypeError as e:
        if logger is not None:
            logger.exception(f"Data {data} was not an integer or could not be cast as an integer.")
        raise FLAMEDtypeError(f"Data {data} was not an integer or could not be cast as an integer.")
    
    try:
        data = _validate_is_greater_than_zero(
            data=data,
            logger=logger,
            accept_nonetype=accept_nonetype
        )
    except FLAMEDtypeError as e:
        if logger is not None:
            logger.exception(f"Data {data} was an integer, but was not greater than 0")
        raise FLAMEDtypeError(f"Data {data} was an integer, but was not positive")

    return data
        

def _float_or_float_array(
        data: Any, 
        logger: Union[Logger, NoneType]=None, 
        dtype: Union[float, np.floating]=float,
        accept_nonetype: bool=False
    ) -> Union[NDArray[np.floating], np.floating, float]:
    """
    If data can be turned into a floating point array, it will.
    If the data can't but matches dtype float, it will return a single integer.
    """
    if data is None and accept_nonetype:
        return data
    
    try:
        proc_data = dtype(data)
    except TypeError as e:
        try:
            proc_data = np.array(data, dtype=dtype)
        except ValueError as e:
            if logger is not None:
                logger.exception(f"Could not convert {data} to float or float array.\nERROR: {e}")
            raise FLAMEDtypeError(f"Could not convert {data} to float or float array.\nERROR: {e}")
    return proc_data


def _validate_is_greater_than_zero(
        data: Union[np.integer, np.floating, NoneType], 
        logger: Union[Logger, NoneType]=None,
        accept_nonetype: bool=True
    ) -> Union[np.integer, np.floating, NoneType]:
    """
    If a number is passed to 'data', this will raise an error if the number is below 1.
    """
    try:
        if not accept_nonetype:
            assert data is not None, f"Data {data} is NoneType, but 'accept_nonetype' is False"
        
        # return None if data is none, otherwise errors ensue
        if data is None:
            return None
        
        assert data > 0, f"Data {data} was not greater than 0"
    except AssertionError as e:
        if logger is not None:
            logger.exception(f"Could not validate {data} is greater than zero.\nERROR: {e}")
        raise FLAMEDtypeError(f"Could not validate {data} is greater than zero.\nERROR: {e}")
    return data


def min_max_norm(
        arr: np.array, 
        mini: Union[np.array, list, int, float], 
        maxi: Union[np.array, list, int, float], 
        sigma: float=1e-20,
        dtype: Union[np.floating]=np.float32
    ) -> NDArray[Union[np.floating]]:
    """
    Min-Max normalized given array based on provided 'mini' and 'maxi'
    If mini and maxi are arrays 
    """
    if (isinstance(mini, Union[np.ndarray, list]) or
        isinstance(maxi, Union[np.ndarray, list])):
        return _min_max_norm_array(
            arr=arr,
            mini=np.array(mini),
            maxi=np.array(maxi),
            sigma=sigma,
            dtype=dtype
        )    
    return (arr - mini) / (maxi - mini + sigma)

def _min_max_norm_array(
        arr: NDArray, 
        mini: NDArray[Union[np.floating, np.integer]], 
        maxi: NDArray[Union[np.floating, np.integer]],
        sigma: float=1e-20,
        dtype: Union[np.floating]=np.float32
    ) -> NDArray[Union[np.floating]]:
    """
    Min-max normalizing based on 'mini' and 'maxi' arrays.
    If 'mini' and 'maxi' are arrays, they must be 1 dimensional, and of equal size.
    The dimension of 'mini' and 'maxi' must match a dimension in the array 'arr'.
    """
    assert mini.ndim == 1 and maxi.ndim == 1
    assert len(mini) == len(maxi)
    assert len(mini) in arr.shape

    axis = list(arr.shape).index(len(mini))
    transpose_arr = []
    for i in range(arr.ndim):
        if i == axis: continue
        transpose_arr.append(i)
    transpose_arr.append(axis)

    arr = arr.transpose(tuple(transpose_arr))
    arr = ((arr - mini) / (maxi - mini + sigma)).astype(dtype)

    new_transpose_arr = []
    for i in range(arr.ndim - 1):
        if i == axis: new_transpose_arr.append(arr.ndim - 1)
        new_transpose_arr.append(i)
    # for the case where the matching dimension is at the end (such as 'ZYXC')
    if len(new_transpose_arr) != arr.ndim: new_transpose_arr.append(arr.ndim - 1)

    return arr.transpose(tuple(new_transpose_arr))


def _apply_bidirectional_correction(img: NDArray, corr: Union[np.integer, int]):
    if corr < 0: # shift leftwards
        img[...,::2,np.abs(corr):] = img[...,::2,:corr]
        img = img[...,np.abs(corr):]
    elif corr > 0: # shift rightwards
        img[...,::2,:-1*corr] = img[...,::2,corr:]
        img = img[...,:-1*corr] # crop image
    else: # case where correction is equal to 0; don't to anything.
        pass
    return img


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except:
        return False
    

def _compress_dict_fields(data: dict) -> dict:
    """
    Description: Convert a many-dimensional dictionary into a single-dimensional dictionary.

    Example:
    my_dict = {
        'A': {
            '1': 'foo',
            '2': {
                'p': 'bar',
                'q': 'lorem'
            }
        },
        'B': 'ipsum'
    }

    _compress_dict_fields(my_dict) -> {
        'A-1': 'foo',
        'A-2-p': 'bar',
        'A-2-q': 'lorem',
        'B': 'ipsum'
    }
    """
    new_data = {}
    for k, v in data.items():
        assert '-' not in k, f"Compression mechanism relies on '-' as separator. Remove '-' from {k} and try again!"
        if isinstance(v, dict):
            sub_data = _compress_dict_fields(v)
            for sub_k, sub_v in sub_data.items():
                new_data[f"{k}-{sub_k}"] = sub_v
        else:
            new_data[k] = v
    return new_data


def _expand_dict_fields(data: dict) -> dict:
    """
    Description: Reverse of _compress_dict_fields().
    """
    new_data = {}
    for k, v in data.items():
        split = k.split('-')
        for s in split[::-1]:
            pass


def set_up_tracking_server(ip: str, port: str, direc: str, log_path: str) -> subprocess.Popen:
    """
    Set up tracking server by spawning up parallel process.

    Args:
     - ip (str): IP where to host the MLFlow server (reocmmend 127.0.0.1 a.k.a. localhost)
     - port (str): Port at IP where to host the MLFlow server
     - direc (str): The directory where mlflow run data & associated artifacts are stored. Typically 'mlruns'
     - log_path (str): Path to directory where mlflow server logs will be stored.

    Returns:
     - proc (subprocess.Popen): Process where the server is being hosted
    """

    server_command = [
        "mlflow", "server",
        "--host", ip,
        "--port", port,
        "--backend-store-uri", direc
    ]
    
    LOGGER.info(f"Starting MLFLOW server with command:\n`{' '.join(server_command)}`")
    MLFLOW_SERVER_LOG = open(log_path, "w+")
    LOGGER.info(f"Starting MLFLOW server log at path {MLFLOW_SERVER_LOG.name}")

    try:
        proc = subprocess.Popen(
            server_command,
            stdout=MLFLOW_SERVER_LOG,
            stderr=MLFLOW_SERVER_LOG
        )
    except Exception as e:
        LOGGER.error(f"Problem starting MLFlow Server process.\n{e.__class__.__name__}: {e}")
        raise FLAMEMLFlowError(f"Problem starting MLFlow Server process.\n{e.__class__.__name__}: {e}")

    return proc


def update_matlab_variables(matlab_eng: str, variable_dict: dict, skip_missing: bool=False) -> None:
    """
    Sync all of the variables in the provided variable dictionaries

    Args:
     - variable_dict (dict): Dictionary of variables to update
     - skip_missing (bool): Whether to skip variables that are not found in MATLAB engine. DEFAULT: False.

    Returns: None. Updates the variables in-place
    """
    for key in variable_dict.keys():
        try:
            variable_dict[key] = matlab_engine.workspace[key]
        except Exception as e:
            if skip_missing:
                LOGGER.error(f"Could not find {key} in ")
        