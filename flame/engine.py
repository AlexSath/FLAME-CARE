import os
import logging
import json

import onnx
import onnxruntime as ort
from onnxruntime import InferenceSession
import numpy as np
from numpy.typing import NDArray

from .image import FLAMEImage
from .error import CAREInferenceError, FLAMEImageError
from .utils import min_max_norm

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
        self.logger = logging.getLogger("main")
        self.execution_providers = None
        self._check_execution_providers(cpu_ok)
        self.model_config = self._load_json(model_config_path)
        self.input_name, self.input_shape, self.input_dtype = None, None, None
        self.inferenceSession = self._load_model(model_path)
        self.dataset_config = self._load_json(dataset_config_path)

        assert 'CUDAExecutionProvider' in ort.get_available_providers()

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
            try:
                try:
                    assert isinstance(image, FLAMEImage), f"Object {image} is not an instance of FLAMEImage."
                    assert image.imShape is not None, f"FLAMEImage {image} was not properly initialized."
                except Exception as e:
                    self.logger.error(f"Could not validate flame image {image}.\nERROR: {e}")
                    raise FLAMEImageError(f"Could not validate flame image {image}.\nERROR: {e}")
                new_list.append(image)
            except FLAMEImageError as e:
                self.logger.warning(f"Skipping image {image}...")
                continue
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
        return self.inferenceSession.run(None, {self.input_name: arr})
    

    def predict_FLAME(self, image: FLAMEImage, input_frames: int=None) -> NDArray:
        """
        Takes FLAMEImage Object and infers on it using the ONNX engine.
        Will attempt to dynamically detect FLAMEImage dimensions (ZFCYX, CYX, etc...)
        and return corresponding denoised image.

        Args:
         - image (FLAMEImage): The FLAMEImage object to be denoised
         - input_frames (int): The number of frames 
        """
        frames = image.get_frames((0, input_frames) if input_frames is not None else None)
        # move channel dimension to the end
        try:
            channel_idx = image.axes_shape.index("C")
        except ValueError as e:
            self.logger.info(f"Could not find channel dimension, so creating one...")
            frames = frames[...,np.newaxis]
            channel_idx = len(image.axes_shape)
        
        if channel_idx != len(image.axes_shape):
            # if channel_idx is already in the last position, no need to transpose
            transpose_shape = []
            for idx in range(len(image.imShape)):
                if idx == channel_idx: continue
                transpose_shape += [idx]
            transpose_shape += [channel_idx]
            frames = np.transpose(frames, tuple(transpose_shape))
    
    def inference_generator(self, inference_images: list[FLAMEImage]):
        """
        Will yield inferred-upon images one-by-one.
        Assumes 1-99 pcttile normalization.
        """
        inference_images = self._validate_FLAME_images(inference_images)

        self.logger.info(f"Inference using 1-99 percentile normalization")
        try:
            input_min = self.dataset_config['FLAME_Dataset']['input']['pixel_1pct']
            input_max = self.dataset_config['FLAME_Dataset']['input']['pixel_99pct']
            self.logger.info(f"Found [{input_min}, {input_max}] for input normalization")
            output_min = self.dataset_config['FLAME_Dataset']['output']['pixel_1pct']
            output_max = self.dataset_config['FLAME_Dataset']['output']['pixel_99pct']
            self.logger.info(f"Found [{output_min}, {output_max}] for output normalization")
        except Exception as e:
            self.logger.error(f"Could not load normalization data from Dataset Config.\nERROR: {e}")
            raise CAREInferenceError(f"Could not load normalization data from Dataset Config.\nERROR: {e}")
        
        for image in inference_images:
            try:
                image.openImage() # loading flame image information into memory
                raw = image.raw()
                raw = np.clip(raw, input_min, input_max)
                raw = min_max_norm(raw, input_min, input_max, dtype=self.input_dtype)
                patches = self._get_patches(image.raw())
                image.closeImage()

                iobinding = self.inferenceSession.io_binding()
                iobinding.bind_cpu_input(self.input_name, patches)
                iobinding.bind_output('output_patches')
                self.inferenceSession.run_with_iobinding(iobinding)
                output_patches = iobinding.copy_outputs_to_cpu()[0]
                output_image = self._stitch_patches(patches=output_patches, final_dim=image.imShape)

                # scale output image to minimum and maximum from dataset config
                output_image = output_image * (np.array(output_max) - np.array(output_min)) + np.array(output_min)

                # rescaling from dataset config can lead to negatives and other foibles, so then rescale to 0.0-1.0
                output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())

            except Exception as e:
                self.logger.error(f"Could not infer on {image}.\nERROR: {e}\nContinuing...")
                continue

            yield output_image


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


