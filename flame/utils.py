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
        