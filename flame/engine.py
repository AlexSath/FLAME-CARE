import os, logging, json, glob, gc
from time import time
from shutil import rmtree
from typing import Union

import onnx
import onnxruntime as ort
from onnxruntime import InferenceSession
import numpy as np
from numpy.typing import NDArray
import mlflow
from mlflow import artifacts

from .image import FLAMEImage, is_FLAME_image
from .error import CAREInferenceError, FLAMEImageError
from .utils import min_max_norm, _float_or_float_array

PATCH_OVERLAP_MAP = {
    16: 4, 32: 8, 64: 16, 128: 32, 256: 64, 512: 128
}

class CAREInferenceSession():
    def __init__(
            self,
            model_path: str, #onnx only, for now
            model_config_path: str,
            dataset_config_path: str,
            cpu_ok: bool=False,
        ) -> None:
        """
        Class CAREInference Session
        
        """
        self.logger = logging.getLogger("ENGINE")
        self.execution_providers = None
        self._check_execution_providers(cpu_ok)
        self.model_config = self._load_json(model_config_path)
        self.input_name, self.input_shape, self.input_dtype = None, None, None
        self.inferenceSession = self._load_model(model_path)
        self.dataset_config = self._load_json(dataset_config_path)
        self.from_mlflow, self.mlflow_tracking_uri = False, None 
        self.mlflow_run_id, self.mlflow_run_name = None, None

        try:
            self.input_min = _float_or_float_array(self.dataset_config['FLAME_Dataset']['input']['pixel_1pct'])
            self.input_max = _float_or_float_array(self.dataset_config['FLAME_Dataset']['input']['pixel_99pct'])
            self.logger.info(f"Found [{self.input_min}, {self.input_max}] for input normalization")
            self.output_min = _float_or_float_array(self.dataset_config['FLAME_Dataset']['output']['pixel_1pct'])
            self.output_max = _float_or_float_array(self.dataset_config['FLAME_Dataset']['output']['pixel_99pct'])
            self.logger.info(f"Found [{self.output_min}, {self.output_max}] for output normalization")
        except Exception as e:
            self.logger.error(f"Could not load normalization data from Dataset Config.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceError(f"Could not load normalization data from Dataset Config.\n{e.__class__.__name__}: {e}")

        if not cpu_ok:
            assert 'CUDAExecutionProvider' in ort.get_available_providers()

    @classmethod
    def from_mlflow_uri(
        cls,
        tracking_uri: str,
        run_id: str,
        model_artifact_path: str="model",
        config_artifact_path: str="model_config",
        model_name: str="model.onnx",
        json_name: str="model_config.json",
        cpu_ok: bool=False
    ):
        logger = logging.getLogger("ENGINE")
        temp_direc = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_direc, exist_ok=True)

        logger.info(f"Loading CAREInferenceSession from MLFlow tracking URI {tracking_uri} and run id {run_id}")
        mlflow.set_tracking_uri(tracking_uri)

        try:
            artifacts.download_artifacts(
                tracking_uri=tracking_uri,
                run_id=run_id,
                artifact_path=model_artifact_path,
                dst_path=temp_direc
            )
        except Exception as e:
            logger.exception(f"Could not load '{model_artifact_path}' path from mlflow run of id {run_id}.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceError(f"Could not load '{model_artifact_path}' path from mlflow run of id {run_id}.\n{e.__class__.__name__}: {e}")
        
        try:
            artifacts.download_artifacts(
                tracking_uri=tracking_uri,
                run_id=run_id,
                artifact_path=config_artifact_path,
                dst_path=temp_direc
            )
        except Exception as e:
            logger.exception(f"Could not load '{config_artifact_path}' path from mlflow run of id {run_id}.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceError(f"Could not load '{config_artifact_path}' path from mlflow run of id {run_id}.\n{e.__class__.__name__}: {e}")
        
        try:
            model_path = glob.glob(os.path.join(temp_direc, "**", model_name), recursive=True)
            json_path = glob.glob(os.path.join(temp_direc, "**", json_name), recursive=True)
            assert len(model_path) == 1 and len(json_path) == 1, f"Expected to found 1 model and one json, not {len(model_path)} and {len(json_path)}, respectively."
            model_path, json_path = model_path[0], json_path[0]
            obj=cls(
                model_path=model_path, 
                model_config_path=json_path,
                dataset_config_path=json_path,
                cpu_ok=cpu_ok
            )
            rmtree(temp_direc)
            setattr(obj, "from_mlflow", True)
            setattr(obj, "mlflow_tracking_uri", tracking_uri)
            setattr(obj, "mlflow_run_id", run_id)
            setattr(obj, "mlflow_run_name", mlflow.get_run(run_id).info.run_name)
            return obj
        except Exception as e:
            logger.exception(f"Could not initialize CAREInferenceSession object.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceError(f"Could not initialize CAREInferenceSession object.\n{e.__class__.__name__}: {e}")


    def _check_execution_providers(self, cpu_ok: bool) -> None:
        """
        Will check for available execution providers.
        Returns None, but sets self.execution_provider.
        """
        try:
            providers = ort.get_available_providers()
            if not cpu_ok: assert "CUDAExecutionProvider" in providers, f"CUDA not available, and 'cpu_ok' is False."
            self.execution_providers = providers
        except Exception as e:
            self.logger.error(f"Could not validate available execution providers with 'cpu_ok' set to {cpu_ok}")
            raise CAREInferenceError(f"Could not validate available execution providers with 'cpu_ok' set to {cpu_ok}")

    def _load_json(self, json_path: str) -> dict:
        try:
            assert os.path.isfile(json_path), f"Provided path {json_path} is not a file."
            json_dict = json.load(open(json_path, 'r'))
        except Exception as e:
            self.logger.error(f"Could not load json from {json_path}.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceError(f"Could not load json from {json_path}.\n{e.__class__.__name__}: {e}")
        return json_dict
    
    def _load_model(self, model_path: str) -> InferenceSession:
        """
        Args:
         - model_path: absolute path to a ".onnx" file
        """
        try:
            assert os.path.isfile(model_path), f"Provided path {model_path} is not a file"
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            del onnx_model
            
            ort_session = ort.InferenceSession(
                model_path,
                providers=self.execution_providers
            )

            input_tensor = ort_session.get_inputs()[0]
            self.input_name, self.input_shape, self.input_dtype = input_tensor.name, input_tensor.shape, input_tensor.type
            self.logger.info(f"Model input: Name-{self.input_name} | Shape-{self.input_shape} | DType-{self.input_dtype}")
        except Exception as e:
            self.logger.error(f"Could not initialize Model Inference Session.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceSession(f"Could not initialize Model Inference Session.\n{e.__class__.__name__}: {e}")
        return ort_session
    
    
    def _validate_FLAME_images(self, inference_images: list) -> dict:
        new_list = []
        for image in inference_images:
            if is_FLAME_image(image): new_list.append(image)
        self.logger.info(f"Of {len(inference_images)} images provided, {len(new_list)} are valid for inference.")
        return new_list
    

    def predict(self, arr: NDArray) -> NDArray:
        """
        Assumes array input of shape NYXC. Will break Y and X dimension into patches necessary for
        inference by the ONNX model in this inference session.

        Args:
         - arr: numpy ndarray of shape NYXC
        
        Returns: Denoised image of shape NYXC
        """
        assert arr.ndim == 4, f"Input array must have 4 dimensions, not {arr.ndim} dimensions of shape {arr.shape}"
        SINGLE_CHANNEL_INFER = True # In the beginning, assume that each channel will be inferred upon one-by-one.
        if self.input_shape[-1] != 1: # if the ONNX input shape is not 1, that means the model was trained for a specific number of channels.
            assert arr.shape[-1] == self.input_shape[-1], f"Array channel dim {arr.shape[-1]} does not match ONNX input channel dim {self.input_shape[-1]} (assumption: NYXC)."
            SINGLE_CHANNEL_INFER = False
            self.logger.info(f"Detected multiple channel inference. Inferring on all channels at the same time.")
        else: self.logger.info(f"Detected single channel inference. Running inference on each channel one-at-a-time.")
        
        input_dim = arr.shape
        input_dtype = arr.dtype
        if SINGLE_CHANNEL_INFER: initial_YXC = list(arr.shape[1:-1]) + [1]
        else: initial_YXC = list(arr.shape[1:])
        
        """NORMALIZATION OPERATIONS"""
        # TODO: Decide what to do if the dataset statistics do not match the number of channels in the input image.
        arr = np.clip(arr, np.array(self.input_min), np.array(self.input_max))
        arr = min_max_norm(arr, np.array(self.input_min), np.array(self.input_max))

        """INFERENCE"""
        def run_on_patches(patches):
            try:
                t1 = time()
                output = self.inferenceSession.run(None, {'patch': patches})[0]
                t2 = time()
                self.logger.info(f"Inference on patches of shape {patches.shape} and dtype {patches.dtype} took {t2 - t1:.2f}s")
                return output
            except Exception as e:
                self.logger.error(f"Inference session failed to predict on patches of dim {patches.shape} and dtype {patches.dtype}")
                raise RuntimeError(f"Inference session failed to predict on patches of dim {patches.shape} and dtype {patches.dtype}")
        
        try:
            full_output = None
            for n in arr: # first looping through all images stacked in the first dimension (N, Y, X, C)
                channel_output = None
                if SINGLE_CHANNEL_INFER:
                    for cdx in range(arr.shape[-1]): # now loopoing through the channel dimensions to predict 1 by 1.
                        patches = self._get_patches(n[...,[cdx]]) # keep channel dimension while indexing
                        output = run_on_patches(patches)
                        output = self._stitch_patches(output, initial_YXC)
                        if channel_output is None: channel_output = output
                        else: channel_output = np.concat([channel_output, output], axis=-1)
                else:
                    patches = self._get_patches(n)
                    channel_output = run_on_patches(patches)
                    channel_output = self._stitch_patches(channel_output, initial_YXC)

                if full_output is None: full_output = channel_output[np.newaxis,...]
                else: full_output = np.concat([full_output, channel_output[np.newaxis,...]], axis=0)
        
        except Exception as e:
            self.logger.error(f"Could not infer on array of shape {input_dim} and dtype {input_dtype}.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceError(f"Could not infer on array of shape {input_dim} and dtype {input_dtype}.\n{e.__class__.__name__}: {e}")
        
        """RENORMALIZATION TO OUTPUT PIXEL DISTRIBUTION"""
        full_output = min_max_norm(full_output, mini=0, maxi=1)
        full_output = (full_output * (self.output_max - self.output_min)) + self.output_min

        """END"""
        return full_output
    

    def predict_FLAME(
            self, 
            image: FLAMEImage, 
            input_frames: int=None,
            # input_min_override,
            # input_max_override,
            # output_min_override,
            # output_max_override
        ) -> NDArray:
        """
        Takes FLAMEImage Object and infers on it using the ONNX engine.
        Will attempt to dynamically detect FLAMEImage dimensions (ZFCYX, CYX, etc...)
        and return corresponding denoised image.

        Args:
         - image (FLAMEImage): The FLAMEImage object to be denoised
         - input_frames (int): The number of frames to input into the denoising model. If none are provided,
                               then all available frames will be used.

        Returns: Numpy NDArray with denoised FLAMEImage data. Will match dimensions of input FLAMEImage.
        """
        
        """ENSURING: Frame and Channel dims exist, getting indicated frames."""
        try:
            frames_idx = image.axes_shape.index("F")
            frames = image.get_frames((0, input_frames) if input_frames is not None else None)
            frame_dim_created = False
        except ValueError as e: # indicates frame dimension was not found.
            # if frame dimension was not found, ensure user did not asked for either 1 frame or all frames
            assert input_frames is None or input_frames == 1, f"User asked for 1 frame or all frames for inference, but no frame dimension was found in {image}"
            # create a frames dimension at the beginning
            frames = image.raw()[np.newaxis,...]
            image.axes_shape = "F" + image.axes_shape
            frame_dim_created = True
        
        # detect where channel dimension is
        try:
            channel_idx = image.axes_shape.index("C")
            channel_dim_created = False
        except ValueError as e:
            self.logger.info(f"Could not find channel dimension, so creating one...")
            frames = frames[...,np.newaxis]
            image.axes_shape = image.axes_shape + "C"
            channel_dim_created = True
            channel_idx = len(image.axes_shape-1)

        """RESHAPING OPERATIONS"""
        # transpose channel dimension to the end
        if channel_idx != len(image.axes_shape) - 1:
            # if channel_idx is already in the last position, no need to transpose
            transpose_shape = []
            for idx in range(frames.ndim):
                if idx == channel_idx: continue
                transpose_shape += [idx]
            transpose_shape += [channel_idx]
            frames = np.transpose(frames, tuple(transpose_shape))

        # Get the current shape. Will be ...,Y,X,C
        original_shape = frames.shape
        # Get the new shape. Will be Z*F*N,Y,X,C
        new_shape = tuple(np.cumprod(frames.shape[:-3])) + (frames.shape[-3:]) # if frames.ndim > 3 else frames.shape
        frames = np.reshape(frames, shape=new_shape)

        """INFERENCE"""
        output_image = self.predict(frames)
        
        """REVERSAL OF RESHAPING OPERATIONS"""
        # recreate the shape of the original frame object.
        output_image = np.reshape(output_image, shape = original_shape)

        # move channel dimension to original position
        new_shape = []
        for adx in range(output_image.ndim - 1):
            if adx == channel_idx:
                new_shape.append(output_image.ndim - 1)
            new_shape.append(adx)
        output_image = np.transpose(output_image, axes=tuple(new_shape))

        """REMOVE ADDED DIMENSIONS"""
        if frame_dim_created:
            image.axes_shape = image.axes_shape[1:]
            output_image = output_image[0,...]

        if channel_dim_created:
            image.axes_shape = image.axes_shape[:-1]
            output_image = output_image[...,0]
        
        """END"""
        return output_image

    
    def inference_generator(self, inference_images: list[FLAMEImage | NDArray], FLAMEImage_input_frames: int=None):
        """
        Will yield inferred-upon images one-by-one.
        Assumes 1-99 pcttile normalization.
        """

        self.logger.info(f"Inference using 1-99 percentile normalization")
        length = len(inference_images)
        res = None
        for idx, image in enumerate(inference_images):
            if is_FLAME_image(image=image):
                try:
                    self.logger.info(f"({idx}/{length}) - Inferring on FLAMEImage {image}...")
                    res = self.predict_FLAME(
                        image=image, # type: ignore
                        input_frames=FLAMEImage_input_frames
                    )
                except Exception as e:
                    self.logger.error(f"FLAMEImage detected, but inference failed.\n{e.__class__.__name__}: {e}")
                    raise CAREInferenceError(f"FLAMEImage detected, but inference failed.\n{e.__class__.__name__}: {e}")
            else:
                try:
                    self.logger.info(f"({idx}/{length}) - Inferring on array (shape: {image.shape} | dtype: {image.dtype})") # type: ignore
                    res = self.predict(
                        arr=image # type: ignore
                    )
                except Exception as e:
                    self.logger.error(f"NDArray detected, but inference failed.\n{e.__class__.__name__}: {e}")
                    raise CAREInferenceError(f"NDArray detected, but inference failed.\n{e.__class__.__name__}: {e}")
        
        if res is None: raise
        return res


    def _get_patch_overlap(self, patch_dim: int) -> int:
        if patch_dim < 16: return PATCH_OVERLAP_MAP[16]
        if patch_dim > 512: return PATCH_OVERLAP_MAP[512]
        try:
            return PATCH_OVERLAP_MAP[patch_dim]
        except KeyError as e:
            key_list = list(PATCH_OVERLAP_MAP)
            for kdx in range(len(PATCH_OVERLAP_MAP)):
                if key_list[kdx] <= patch_dim:
                    return key_list[kdx]
            else:
                raise

    def _get_patches(self, arr: NDArray) -> NDArray:
        """
        Description: _get_patches will break down an input image (as an NDArray)
        into patches that can be inferred upon. Patch dimensions will be that of the
        listed self.model_config -> 'patch_size' -> "Patch_Config".

        Args:
         - arr: A numpy NDArray. Should have dimensions YXC

        Returns: 
         - A numpy NDArray array of dimensions (N, patch_size, patch_size, C).
        """
        try:
            patch_dim = self.model_config["Patch_Config"]['patch_size']
        except Exception as e:
            self.logger.error(f"Could not retrieve patch dimensions from self.model_config.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceError(f"Could not retrieve patch dimensions from self.model_config.\n{e.__class__.__name__}: {e}")
        
        try:
            po = self._get_patch_overlap(patch_dim=patch_dim)
        except Exception as e:
            self.logger.error(f"Could not retrieve patch overlap.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceError(f"Could not retrieve patch overlap.\n{e.__class__.__name__}: {e}")

        try:
            assert len(arr.shape) == 3, f"Input dimensions must be of size 3 (YXC), not {len(arr.shape)}."
            input_y, input_x, input_c = arr.shape
        except Exception as e:
            self.logger.error(f"Cannot interpret input dimensions for patch extraction.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceError(f"Cannot interpret input dimensions for patch extraction.\n{e.__class__.__name__}: {e}")
        
        # if input array is the size of the input patch, just return it with new batch dimension (a.k.a. N)
        if patch_dim == input_y and patch_dim == input_x: return arr[np.newaxis, ...]

        # NOTE: This will still break if ONE of input arary dimensions matches the patch dimension, but the other doesn't.
        # See issue #8 https://github.com/AlexSath/BaluLab-CARE/issues/8

        output = None
        start_x = 0
        start_y = 0
        while start_y + patch_dim < input_y:
            while start_x + patch_dim < input_x:
                if start_y == 0 and start_x == 0: # top left corner
                    this_patch = arr[start_y:start_y+patch_dim, start_x:start_x+patch_dim, :]
                    assert this_patch.shape == (128, 128, 1), f"{this_patch.shape}"
                elif start_y == 0 and start_x != 0: # top of image
                    this_patch = arr[start_y:start_y+patch_dim, start_x-po//2:start_x-po//2+patch_dim, :]
                    assert this_patch.shape == (128, 128, 1), f"{this_patch.shape}"
                elif start_y != 0 and start_x == 0: # left side of image (leftmost column, any y)
                    this_patch = arr[start_y-po//2:start_y+patch_dim-po//2, start_x:start_x+patch_dim, :]
                    assert this_patch.shape == (128, 128, 1), f"{this_patch.shape}"
                else: # center of image
                    this_patch = arr[start_y-po//2:start_y+patch_dim-po//2, start_x-po//2:start_x+patch_dim-po//2, :]
                    assert this_patch.shape == (128, 128, 1), f"{this_patch.shape}"
                if output is None: output = [this_patch]
                else: output += [this_patch]
                start_x += patch_dim - po
            
            # if x's don't go evenly into input dimension, then run code to ensure right of image is denoised..
            if input_x % patch_dim != 0: # rightmost column
                if start_y == 0: # top right corner
                    this_patch = arr[start_y:start_y+patch_dim, -patch_dim:, :]
                    assert this_patch.shape == (128, 128, 1), f"{this_patch.shape}"
                else: # right side
                    this_patch = arr[start_y-po//2:start_y+patch_dim-po//2, -patch_dim:, :]
                    assert this_patch.shape == (128, 128, 1), f"{this_patch.shape}"
                output += [this_patch]

            start_x = 0
            start_y += patch_dim - po

        # If y's don't go evenly into input dimension, then run code to ensure bottom of image is denoised.
        if input_y % patch_dim != 0:
            while start_x + patch_dim < input_x:
                if start_x == 0: # bottom left corner 
                    this_patch = arr[-patch_dim:, start_x:start_x+patch_dim, :]
                    assert this_patch.shape == (128, 128, 1), f"{this_patch.shape}"
                else: # bottom side
                    this_patch = arr[-patch_dim:, start_x-po//2:start_x+patch_dim-po//2, :]
                    assert this_patch.shape == (128, 128, 1), f"{this_patch.shape}"
                output += [this_patch]
                start_x += patch_dim - po
            
            if input_x % patch_dim != 0: # bottom right corner
                this_patch = arr[-patch_dim:, -patch_dim:, :]
                assert this_patch.shape == (128, 128, 1), f"{this_patch.shape}"
                output += [this_patch]

        try:
            assert output is not None, f"Output is NoneType. Not patches could be extracted. Check dimensions of image {arr.shape} and patch ({patch_dim}, {patch_dim}, C)"
            assert len(output) > 0, f"No patches could be extracted from input image of shape {arr.shape} and dtype {arr.dtype}"
            if len(output) == 1: return output[0][np.newaxis,...]
            else: return np.stack(output, axis=0)
        except Exception as e:
            self.logger.error(f"Could not output extracted patches.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceError(f"Could not output extracted patches.\n{e.__class__.__name__}: {e}")


    def _stitch_patches(self, patches: NDArray, final_dim: tuple[int, int, int]) -> NDArray:
        """
        Description: _stitch_patches will take a patch array of shape (N, patch_y, patch_x, C)
        and stitch it back into a full-size image of shape 'final_dim'.

        Args:
         - patches: numpy NDArray of shape (N, patch_y, patch_x, C)
         - final_dim: final dimensions of the image. Should match axes YXC. C in final_dim
                      should match C in the dimension of 'patches'.
        
        """
        try:
            input_y, input_x, input_c = final_dim
        except Exception as e:
            self.logger.error(f"Could not inppack 'final_dim' of size {len(final_dim)} into Y,X,C.\n{e.__class__.__name__} {e}")
            raise CAREInferenceError(f"Could not inppack 'final_dim' of size {len(final_dim)} into Y,X,C.\n{e.__class__.__name__} {e}")
        
        # get patch dimension
        assert patches.shape[-1] == input_c, f"Channels in patch array and final_dim do not match ({patches.shape[-1]} vs. {input_c})"
        assert patches.shape[1] == patches.shape[2], f"Rectangular patch detected (assuming axes NYXC). Only square patches are supported"
        patch_dim = patches.shape[1]

        try: # get patch overlap
            po = self._get_patch_overlap(patch_dim=patch_dim)
        except Exception as e:
            self.logger.error(f"Could not retrieve patch overlap.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceError(f"Could not retrieve patch overlap.\n{e.__class__.__name__}: {e}")
        

        try:
            patch_overlap = self._get_patch_overlap(patch_dim=patch_dim)
        except Exception as e:
            self.logger.error(f"Could not retrieve patch overlap.\n{e.__class__.__name__}: {e}")
            raise CAREInferenceError(f"Could not retrieve patch overlap.\n{e.__class__.__name__}: {e}")

        output = np.zeros(shape=final_dim, dtype=patches.dtype)
        this_y = 0
        this_x = 0
        patch_index = 0
        while this_y + patch_dim < input_y:
            while this_x + patch_dim < input_x:
                # output[this_y:this_y+patch_dim, this_x:this_x+patch_dim, :] = patches[patch_index,...]
                if this_x == 0 and this_y == 0: # top left corner
                    output[this_y:this_y+patch_dim-po//2, this_x:this_x+patch_dim-po//2] = patches[patch_index, :-po//2, :-po//2, :] # crop bottom and right of patch
                elif this_x == 0 and this_y != 0: # left side
                    output[this_y:this_y+patch_dim-po, this_x:this_x+patch_dim-po//2] = patches[patch_index, po//2:-po//2, :-po//2, :] # crop top, bottom and right of patch
                elif this_x != 0 and this_y ==0: # top of image
                    output[this_y:this_y+patch_dim-po//2, this_x:this_x+patch_dim-po] = patches[patch_index, :-po//2, po//2:-po//2, :]
                else: # middle of image in both axes
                    output[this_y:this_y+patch_dim-po, this_x:this_x+patch_dim-po] = patches[patch_index, po//2:-po//2, po//2:-po//2, :] # crop all sides of patch
                patch_index += 1
                this_x += patch_dim - po

            # if patch size doesn't go evenly into x axis, ensure that right edge of image is still stitched
            if input_x % patch_dim != 0:
                # output[this_y:this_y+patch_dim, -patch_dim:, :] = patches[patch_index,...]
                if this_y == 0: # top right corner
                    output[this_y:this_y+patch_dim-po//2, -patch_dim+po//2:, :] = patches[patch_index,:-po//2,po//2:,:] # crop bottom and left of patch
                else: # right edge; middle of y axis (right side of image)
                    output[this_y:this_y+patch_dim-po, -patch_dim+po//2:, :] = patches[patch_index,po//2:-po//2,po//2:] # crop top, bottom, and left of patch
                patch_index += 1

            this_x = 0
            this_y += patch_dim - po

        # bottom row (if patch size doesn't go evenly into y axis)
        if input_y % patch_dim != 0:
            while this_x + patch_dim < input_x:
                # output[-patch_dim:, this_x:this_x+patch_dim, :] = patches[patch_index,...]
                if this_x == 0: # bottom left corner
                    output[-patch_dim+po//2:, this_x:this_x+patch_dim-po//2, :] = patches[patch_index,po//2:,:-po//2,:] # crop top and right of patch
                else:
                    output[-patch_dim+po//2:, this_x:this_x+patch_dim-po, :] = patches[patch_index,po//2:,po//2:-po//2,:] # crop top, left, and right of patch
                patch_index += 1
                this_x += patch_dim - po

            # last corner (if patch size doesn't go evenly into x axis)
            if input_x % patch_dim != 0:
                # output[-patch_dim:, -patch_dim:, :] = patches[patch_index,...]
                output[-patch_dim+po//2:, -patch_dim+po//2:, :] = patches[patch_index,po//2:,po//2:,:] # crop top and left of patch
                patch_index += 1

        return output
    
    def __repr__(self):
        str = f"Obj CAREInferenceSession @{hex(id(self))}:\n" \
         + f" - Input Dim: {self.input_shape}\n" \
         + f" - Input DType: {self.input_dtype}\n" \
         + f" - From MLFlow: {self.from_mlflow}\n"

        if self.from_mlflow:
            str += f"" \
             + f" - MLFlow Tracking URI: {self.mlflow_tracking_uri}\n" \
             + f" - MLFlow Run ID: {self.mlflow_run_id}\n" \
             + f" - MLFlow Run Name: {self.mlflow_run_name}\n"

        return str
    
    def __str__(self):
        return repr(self)


