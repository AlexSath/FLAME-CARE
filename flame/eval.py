import logging

from skimage.metrics import structural_similarity
import numpy as np

from .error import FLAMEEvalError

logger = logging.getLogger("Eval")

def mae(a1, a2):
    try:
        assert a1.shape == a2.shape, f"Inputs must have the same shape, not {a1.shape} and {a2.shape}."
        assert a1.dtype == a2.dtype, f"Inputs must ahve the same dtype, not {a1.dtype} and {a2.dtype}."
        return np.mean(np.abs(a1 - a2))
    except Exception as e:
        logger.error(f"Could not calculate MAE from inputs.\nEXCEPTION: {e}")
        raise FLAMEEvalError(f"Could not calculate MAE from inputs.\nEXCEPTION: {e}")


def mse(a1, a2):
    try:
        assert a1.shape == a2.shape, f"Inputs must have the same shape, not {a1.shape} and {a2.shape}."
        assert a1.dtype == a2.dtype, f"Inputs must ahve the same dtype, not {a1.dtype} and {a2.dtype}."
        return np.mean((a1 - a2).astype(np.float128)**2)
    except Exception as e:
        logger.error(f"Could not calculate MSE from inputs.\nEXCEPTION: {e}")
        raise FLAMEEvalError(f"Could not calculate MSE from inputs.\nEXCEPTION: {e}")

def ssim(a1, a2, channel_axis=2):
    try:
        assert a1.shape == a2.shape, f"Inputs must have the same shape, not {a1.shape} and {a2.shape}."
        assert a1.dtype == a2.dtype, f"Inputs must ahve the same dtype, not {a1.dtype} and {a2.dtype}."
        return structural_similarity(
            im1=a1,
            im2=a2,
            data_range=np.max([np.max(a1), np.max(a2)]),
            channel_axis=channel_axis,
            gradient=False, # means that the gradient SSIM will not be returned
            full=False, # means that the full SSIM image will not be returned
        )
    except Exception as e:
        logger.error(f"Could not calculate SSIM from inputs.\nEXCEPTION: {e}")
        raise FLAMEEvalError(f"Could not calculate SSIM from inputs.\nEXCEPTION: {e}")