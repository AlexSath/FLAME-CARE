from logging import Logger
from typing import Union, Any
from types import NoneType

import numpy as np
from numpy.typing import NDArray

from .error import FLAMEDtypeError

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



def min_max_norm(arr: np.array, mini: Union[np.array, int, float], maxi: Union[np.array, int, float], sigma: float=1e-20) -> np.array:
    """
    Min-Max normalized given array based on provided 'mini' and 'maxi'
    """
    if type(mini) == np.array:
        assert mini.ndim == 1, f"Expected array of shape (N, ) not shape {mini.shape}"
        assert mini.shape == arr.shape or mini.shape[0] == arr.shape[-1], f"If mini/maxi are np.array, their Dim should match Arr"
    if type(maxi) == np.array:
        assert maxi.ndim == 1, f"Expected array of shape (N, ) not shape {maxi.shape}"
        assert maxi.shape == arr.shape or maxi.shape[0] == arr.shape[-1], f"If mini/maxi are np.array, their Dim should match Arr"
    assert arr.ndim <= 3, f"Input array should have 3 or fewer dimensions, not shape {arr.shape}"

    return (arr - mini) / (maxi - mini + sigma)