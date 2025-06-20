import os, logging, json, glob
from shutil import rmtree
from typing import Union

import onnx
import onnxruntime as ort
from onnxruntime import InferenceSession
import numpy as np
from numpy.typing import NDArray
import mlflow.artifacts as artifacts

from .image import FLAMEImage, is_FLAME_image
from .error import CAREInferenceError, FLAMEImageError
from .utils import min_max_norm, _float_or_float_array

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
        self.logger = logging.getLogger("CareInference")
        self.execution_providers = None
        self._check_execution_providers(cpu_ok)
        self.model_config = self._load_json(model_config_path)
        self.input_name, self.input_shape, self.input_dtype = None, None, None
        self.inferenceSession = self._load_model(model_path)
        self.dataset_config = self._load_json(dataset_config_path)
        self.input_min, self.input_max = None, None
        self.output_min, self.output_max = None, None

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
        logger = logging.getLogger("CareInference")
        temp_direc = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_direc, exist_ok=True)

        try:
            artifacts.download_artifacts(
                tracking_uri=tracking_uri,
                run_id=run_id,
                artifact_path=model_artifact_path,
                dst_path=temp_direc
            )
        except Exception as e:
            logger.error(f"Could not load '{model_artifact_path}' path from mlflow run of id {run_id}.\nEXCEPTION: {e}")
            raise CAREInferenceError(f"Could not load '{model_artifact_path}' path from mlflow run of id {run_id}.\nEXCEPTION: {e}")
        
        try:
            artifacts.download_artifacts(
                tracking_uri=tracking_uri,
                run_id=run_id,
                artifact_path=config_artifact_path,
                dst_path=temp_direc
            )
        except Exception as e:
            logger.error(f"Could not load '{config_artifact_path}' path from mlflow run of id {run_id}.\nEXCEPTION: {e}")
            raise CAREInferenceError(f"Could not load '{config_artifact_path}' path from mlflow run of id {run_id}.\nEXCEPTION: {e}")
        
        try:
            model_path = glob.glob(os.path.join(temp_direc, "**", model_name), recursive=True)
            json_path = glob.glob(os.path.join(temp_direc, "**", json_name), recursive=True)
            assert len(model_path) == 1 and len(json_path) == 1, f"Expected to found 1 model and one json, not {len(model_path)} and {len(json_path)}, respectively."
            model_path, json_path = model_path[0], json_path[0]
            obj=cls(
                model_path=model_path, 
                model_config_path=json_path,
                dataset_config_path=json_path,
            )
            rmtree(temp_direc)
            return obj
        except Exception as e:
            logger.error(f"Could not initialize CAREInferenceSession object.\nEXCEPTION: {e}")
            raise CAREInferenceError(f"Could not initialize CAREInferenceSession object.\nEXCEPTION: {e}")


    def _check_execution_providers(self, cpu_ok: bool) -> None:
        """
        Will check for available execution providers.
        Returns None, but sets self.execution_provider.
        """
        try:
            providers = ort.get_available_providers()
            if not cpu_ok: assert "CUDAExecutionProvider" in providers, f"CUDA not available, and 'cpu_ok' is False."
            else: self.execution_providers = providers
        except Exception as e:
            self.logger.error(f"Could not validate available execution providers with 'cpu_ok' set to {cpu_ok}")
            raise CAREInferenceError(f"Could not validate available execution providers with 'cpu_ok' set to {cpu_ok}")

    def _load_json(self, json_path: str) -> dict:
        try:
            assert os.path.isfile(json_path), f"Provided path {json_path} is not a file."
            json_dict = json.load(open(json_path, 'r'))
        except Exception as e:
            self.logger.error(f"Could not load json from {json_path}.\nERROR: {e}")
            raise CAREInferenceError(f"Could not load json from {json_path}.\nERROR: {e}")
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
            self.logger.error(f"Could not initialize Model Inference Session.\nERROR: {e}")
            raise CAREInferenceSession(f"Could not initialize Model Inference Session.\nERROR: {e}")
        return ort_session
    
    
    def _validate_FLAME_images(self, inference_images: list) -> dict:
        new_list = []
        for image in inference_images:
            if is_FLAME_image(image): new_list.append(image)
        self.logger.info(f"Of {len(inference_images)} images provided, {len(new_list)} are valid for inference.")
        return new_list
    

    def predict(self, arr: NDArray) -> NDArray:
        """
        Assumes array input of shape YXC or NYXC. Also assumes that input tensor to ONNX model
        will be of shape (None, Y, X, C)

        Args:
         - arr: numpy ndarray of shape YXC or NYXC. Y and X dimensions should match Y an X of ONNX model input tensor.
        
        Returns: ONNX Model output
        """
        # TODO: pre-inference normalization and post inference return to range.
        try:
            output = []
            for cdx in range(arr.shape[-1]):
                this_out = self.inferenceSession.run(None, {self.input_name: arr[...,[cdx]]})
                output.append(this_out)
            return np.stack(output, axis=-1)
        except Exception as e:
            self.logger.error(f"Could not infer on array of shape {arr.shape} and dtype {arr.dtype}.\nERROR: {e}")
            raise CAREInferenceError(f"Could not infer on array of shape {arr.shape} and dtype {arr.dtype}.\nERROR: {e}")
    

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
        """

        try:
            if self.input_min is None or self.input_max is None or self.output_min is None or self.output_max is None:
                self.input_min = _float_or_float_array(self.dataset_config['FLAME_Dataset']['input']['pixel_1pct'])
                self.input_max = _float_or_float_array(self.dataset_config['FLAME_Dataset']['input']['pixel_99pct'])
                self.logger.info(f"Found [{self.input_min}, {self.input_max}] for input normalization")
                self.output_min = _float_or_float_array(self.dataset_config['FLAME_Dataset']['output']['pixel_1pct'])
                self.output_max = _float_or_float_array(self.dataset_config['FLAME_Dataset']['output']['pixel_99pct'])
                self.logger.info(f"Found [{self.output_min}, {self.output_max}] for output normalization")
        except Exception as e:
            self.logger.error(f"Could not load normalization data from Dataset Config.\nERROR: {e}")
            raise CAREInferenceError(f"Could not load normalization data from Dataset Config.\nERROR: {e}")
        
        try:
            frames_idx = image.axes_shape.index("F")
            frames = image.get_frames((0, input_frames) if input_frames is not None else None)
            frame_dim_created = False
        except ValueError as e: # indicates frame dimension was not found.
            # create a frames dimension at the beginning
            frames = image.raw()[np.newaxis,...]
            image.axes_shape = "F" + image.axes_shape
            frame_dim_created = True
        
        """RESHAPING OPERATIONS"""
        # detect where channel dimension is
        try:
            channel_idx = image.axes_shape.index("C")
        except ValueError as e:
            self.logger.info(f"Could not find channel dimension, so creating one...")
            frames = frames[...,np.newaxis]
            image.axes_shape = image.axes_shape + "C"
            channel_idx = len(image.axes_shape-1)
        
        # transpose channel dimension to the end
        if channel_idx != len(image.axes_shape) - 1:
            # if channel_idx is already in the last position, no need to transpose
            transpose_shape = []
            for idx in range(frames.ndim):
                if idx == channel_idx: continue
                transpose_shape += [idx]
            transpose_shape += [channel_idx]
            print(transpose_shape)
            frames = np.transpose(frames, tuple(transpose_shape))

        # Get the current shape. Will be ...,Y,X,C
        original_shape = frames.shape
        # Get the new shape. Will be Z*F*N,Y,X,C
        new_shape = tuple(np.cumprod(frames.shape[:-3])) + (frames.shape[-3:]) # if frames.ndim > 3 else frames.shape
        print(new_shape)
        frames = np.reshape(frames, shape=new_shape)

        """NORMALIZATION OPERATIONS"""
        # TODO: Decide what to do if the dataset statistics do not match the number of channels in the input image.
        frames = np.clip(frames, np.array(self.input_min), np.array(self.input_max))
        frames = min_max_norm(frames, np.array(self.input_min), np.array(self.input_max))

        """INFERENCE"""
        # Getting the output
        full_output = []
        for n in frames: # first looping through all images stacked in the first dimension (N, Y, X, C)
            channel_output = []
            for cdx in range(frames.shape[-1]): # now loopoing through the channel dimensions to predict 1 by 1.
                patches = self._get_patches(n[...,[cdx]]) # keep dimension while indexing
                output = self.predict(patches)
                channel_output.append(self._stitch_patches(output, image._get_dims(axes="YXC")))
            full_output.append(np.stack(channel_output, axis=-1))

        """REVERSAL OF RESHAPING OPERATIONS"""
        # recreate the shape of the original frame object.
        output_image = np.stack(full_output, axis=0)
        output_image = np.reshape(output_image, shape = original_shape)

        # move channel dimension to original position
        new_shape = []
        for adx in range(output_image.ndim - 1):
            if adx == channel_idx:
                new_shape.append(output_image.ndim - 1)
            new_shape.append(adx)
        output_image = np.transpose(output_image, axes=tuple(new_shape))

        """REVERSAL OF NORMALIZATION OPERATIONS"""
        # scale output image to minimum and maximum from dataset config
        output_image = output_image * (np.array(self.output_max) - np.array(self.output_min)) + np.array(self.output_min)
        # rescaling from dataset config can lead to negatives and other foibles, so then rescale to 0.0-1.0
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())

        """END"""
        if frame_dim_created:
            image.axes_shape = image.axes_shape[1:]
            return output_image[0,...]
        else:
            return output_image

    
    def inference_generator(self, inference_images: list[FLAMEImage | NDArray], FLAMEImage_input_frames: int=None):
        """
        Will yield inferred-upon images one-by-one.
        Assumes 1-99 pcttile normalization.
        """

        self.logger.info(f"Inference using 1-99 percentile normalization")
        length = len(inference_images)
        for idx, image in enumerate(inference_images):
            if is_FLAME_image(image):
                try:
                    self.logger.info(f"({idx}/{length}) - Inferring on FLAMEImage {image}...")
                    res = self.predict_FLAME(
                        image=image, 
                        input_frames=FLAMEImage_input_frames
                    )
                except Exception as e:
                    self.logger.error(f"FLAMEImage detected, but inference failed.\nERROR: {e}")
                    raise CAREInferenceError(f"FLAMEImage detected, but inference failed.\nERROR: {e}")
            else:
                try:
                    self.logger.info(f"({idx}/{length}) - Inferring on array (shape: {image.shape} | dtype: {image.dtype})")
                    res = self.predict(
                        arr=image
                    )
                except Exception as e:
                    self.logger.error(f"NDArray detected, but inference failed.\nERROR: {e}")
                    raise CAREInferenceError(f"NDArray detected, but inference failed.\nERROR: {e}")
        
        return res


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
            self.logger.error(f"Could not retrieve patch dimensions from self.model_config.\nERROR: {e}")
            raise CAREInferenceError(f"Could not retrieve patch dimensions from self.model_config.\nERROR: {e}")

        try:
            assert len(arr.shape) == 3, f"Input dimensions must be of size 3 (YXC), not {len(arr.shape)}."
            input_y, input_x, input_c = arr.shape
        except Exception as e:
            self.logger.error(f"Cannot interpret input dimensions for patch extraction.\nERROR: {e}")
            raise CAREInferenceError(f"Cannot interpret input dimensions for patch extraction.\nERROR: {e}")

        output = None
        start_x = 0
        start_y = 0
        while start_y + patch_dim < input_y:
            while start_x + patch_dim < input_x:
                this_patch = arr[start_y:start_y+patch_dim, start_x:start_x+patch_dim, :]
                if output is None: output = [this_patch]
                else: output += [this_patch]
                start_x += patch_dim
            
            # if x's don't go evenly into input dimension, then run code to ensure right of image is denoised..
            if input_x % patch_dim != 0:
                output += [arr[start_y:start_y+patch_dim, -patch_dim:, :]]

            start_x = 0
            start_y += patch_dim

        # If y's don't go evenly into input dimension, then run code to ensure bottom of image is denoised.
        if input_y % patch_dim != 0:
            while start_x + patch_dim < input_x:
                this_patch = arr[-patch_dim:, start_x:start_x+patch_dim, :]
                output += [this_patch]
                start_x += patch_dim
            
            if input_x % patch_dim != 0:
                output += [arr[-patch_dim:, -patch_dim:, :]]

        return np.stack(output, axis=0)


    def _stitch_patches(self, patches: NDArray, final_dim: tuple[int]) -> NDArray:
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
            self.logger.error(f"Could not inppack 'final_dim' of size {len(final_dim)} into Y,X,C.\nERROR {e}")
            raise CAREInferenceError(f"Could not inppack 'final_dim' of size {len(final_dim)} into Y,X,C.\nERROR {e}")
        
        assert patches.shape[-1] == input_c, f"Channels in patch array and final_dim do not match ({patches.shape[-1]} vs. {input_c})"

        assert patches.shape[1] == patches.shape[2], f"Rectangular patch detected (assuming axes NYXC). Only square patches are supported"
        patch_dim = patches.shape[1]

        output = np.zeros(shape=final_dim, dtype=patches.dtype)
        this_y = 0
        this_x = 0
        patch_index = 0
        while this_y + patch_dim < final_dim[0]:
            while this_x + patch_dim < final_dim[1]:
                output[this_y:this_y+patch_dim, this_x:this_x+patch_dim, :] = patches[patch_index,...]
                patch_index += 1
                this_x += patch_dim

            if final_dim[1] % patch_dim != 0:
                output[this_y:this_y+patch_dim, -patch_dim:, :] = patches[patch_index,...]
                patch_index += 1

            this_x = 0
            this_y += patch_dim

        if final_dim[0] % patch_dim != 0:
            while this_x + patch_dim < final_dim[1]:
                output[-patch_dim:, this_x:this_x+patch_dim, :] = patches[patch_index,...]
                patch_index += 1
                this_x += patch_dim

            if final_dim[1] % patch_dim != 0:
                output[-patch_dim:, -patch_dim:, :] = patches[patch_index,...]
                patch_index += 1

        return output


