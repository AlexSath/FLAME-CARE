from typing import Union

import numpy as np

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