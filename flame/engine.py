import os
import logging
import json

import onnx
import onnxruntime as ort
from onnxruntime import InferenceSession

from .image import FLAMEImage
from .error import CAREInferenceError, FLAMEImageError

class CAREInferenceSession():
    def __init__(
            self,
            model_path: str, #onnx only, for now
            model_config_path: str,
            dataset_config_path: str,
            inference_images: list,
            cpu_ok: bool
        ) -> None:
        """
        Class CAREInference Session
        
        """
        self.logger = logging.getLogger("main")
        self.execution_providers = None
        self._check_execution_providers(cpu_ok)
        self.model_config = self._load_json(model_config_path)
        self.input_name, self.input_shape, self.input_dtype = None, None, None
        self.inference_session = self._load_model(model_path)
        self.dataset_config = self._load_json(dataset_config_path)
        self.inference_images = self._validate_FLAME_images(inference_images)

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

