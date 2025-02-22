from enum import Enum
from typing import List

import capybara as cb
import numpy as np

from .det import Inference as DetectionInference
from .rec import Inference as RecognitionInference
from .spotting import Inference as SpottingInference
from .utils import replace_digits, replace_letters, replace_sex

__all__ = [
    'MRZScanner', 'ModelType', 'SpottingInference', 'ErrorCodes']


class ModelType(cb.EnumCheckMixin, Enum):
    spotting = 1
    two_stage = 2
    detection = 3
    recognition = 4


class ErrorCodes(Enum):
    NO_ERROR = 'No error.'
    INVALID_INPUT_FORMAT = 'Invalid input format.'
    POSTPROCESS_FAILED_LINE_COUNT = 'Postprocess failed, number of lines not 2 or 3.'
    POSTPROCESS_FAILED_TD1_LENGTH = 'Postprocess failed, length of lines not 30 when `doc_type` is TD1.'
    POSTPROCESS_FAILED_TD2_TD3_LENGTH = 'Postprocess failed, length of lines not 36 or 44 when `doc_type` is TD2 or TD3.'


class MRZScanner:

    def __init__(
        self,
        model_type: ModelType = ModelType.two_stage,
        model_cfg: str = None,
        spotting_cfg: str = None,
        detection_cfg: str = None,
        recognition_cfg: str = None,
        backend: cb.Backend = cb.Backend.cpu,
        gpu_id: int = 0,
        **kwargs
    ) -> None:
        """ Initialize MRZScanner.

        Args:
            model_type (ModelType): Model type.
            model_cfg (str): Default model configuration (used when specific configs are not provided).
            spotting_cfg (str): Spotting model configuration.
            detection_cfg (str): Detection model configuration.
            recognition_cfg (str): Recognition model configuration.
            backend (cb.Backend): Backend.
            gpu_id (int): GPU ID.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If invalid model_cfg is provided.
        """

        self.scanner = None
        self.detector = None
        self.recognizer = None

        self.model_type = ModelType.obj_to_enum(model_type)

        # Assign model configurations with fallback to general model_cfg
        spotting_cfg = spotting_cfg or model_cfg or '20240919'
        detection_cfg = detection_cfg or model_cfg or '20250222'
        recognition_cfg = recognition_cfg or model_cfg or '20250221'

        if self.model_type == ModelType.spotting:
            self._init_spotting(spotting_cfg, gpu_id, backend, **kwargs)
        elif self.model_type == ModelType.detection:
            self._init_detection(detection_cfg, gpu_id, backend, **kwargs)
        elif self.model_type == ModelType.recognition:
            self._init_recognition(recognition_cfg, gpu_id, backend, **kwargs)
        elif self.model_type == ModelType.two_stage:
            self._init_detection(detection_cfg, gpu_id, backend, **kwargs)
            self._init_recognition(recognition_cfg, gpu_id, backend, **kwargs)
        else:
            raise ValueError(
                f'Invalid model_type: {model_type}, valid model_types: {list(ModelType)}'
            )

    def _init_spotting(self, model_cfg, gpu_id, backend, **kwargs):
        valid_model_cfgs = list(SpottingInference.configs.keys())
        if model_cfg not in valid_model_cfgs:
            raise ValueError(
                f'Invalid spotting_cfg: {model_cfg}, valid configs: {valid_model_cfgs}'
            )
        self.scanner = SpottingInference(
            gpu_id=gpu_id,
            backend=backend,
            model_cfg=model_cfg,
            **kwargs
        )

    def _init_detection(self, model_cfg, gpu_id, backend, **kwargs):
        valid_model_cfgs = list(DetectionInference.configs.keys())
        if model_cfg not in valid_model_cfgs:
            raise ValueError(
                f'Invalid detection_cfg: {model_cfg}, valid configs: {valid_model_cfgs}'
            )
        self.detector = DetectionInference(
            gpu_id=gpu_id,
            backend=backend,
            model_cfg=model_cfg,
            **kwargs
        )

    def _init_recognition(self, model_cfg, gpu_id, backend, **kwargs):
        valid_model_cfgs = list(RecognitionInference.configs.keys())
        if model_cfg not in valid_model_cfgs:
            raise ValueError(
                f'Invalid recognition_cfg: {model_cfg}, valid configs: {valid_model_cfgs}'
            )
        self.recognizer = RecognitionInference(
            gpu_id=gpu_id,
            backend=backend,
            model_cfg=model_cfg,
            **kwargs
        )

    def list_models(self) -> List[str]:
        spotting_models = list(SpottingInference.configs.keys())
        detection_models = list(DetectionInference.configs.keys())
        recognition_models = list(RecognitionInference.configs.keys())
        infos = {
            'spotting': spotting_models,
            'detection': detection_models,
            'recognition': recognition_models
        }
        return infos

    def postprocess(self, results: np.ndarray) -> List[str]:
        if (doc_type := len(results)) not in [2, 3]:
            return [''], ErrorCodes.POSTPROCESS_FAILED_LINE_COUNT

        if doc_type == 3:  # TD1
            if len(results[0]) != 30 or len(results[1]) != 30 or len(results[2]) != 30:
                return [''], ErrorCodes.POSTPROCESS_FAILED_TD1_LENGTH
            # Line1
            doc = results[0][0:2]
            country = replace_digits(results[0][2:5])
            doc_number = results[0][5:14]
            doc_number_hash = replace_letters(results[0][14])
            optional = results[0][15:30]
            results[0] = f'{doc}{country}{doc_number}{doc_number_hash}{optional}'
            # Line2
            birth_date = replace_letters(results[1][0:7])
            sex = replace_sex(results[1][7])
            expiry_date = replace_letters(results[1][8:15])
            nationality = replace_digits(results[1][15:18])
            optional = results[1][18:30]
            results[1] = f'{birth_date}{sex}{expiry_date}{nationality}{optional}'
            return results, ErrorCodes.NO_ERROR

        elif doc_type == 2:  # TD2 or TD3

            if not ((len(results[0]) == 36 and len(results[1]) == 36) or
                    (len(results[0]) == 44 and len(results[1]) == 44)):
                return [''], ErrorCodes.POSTPROCESS_FAILED_TD2_TD3_LENGTH

            # Line2
            doc_number = results[1][0:9]
            doc_number_hash = replace_letters(results[1][9])
            nationality = replace_digits(results[1][10:13])
            birth_date = replace_letters(results[1][13:20])
            sex = replace_sex(results[1][20])
            expiry_date = replace_letters(results[1][21:28])
            optional = results[1][28:]
            results[1] = f'{doc_number}{doc_number_hash}{nationality}{birth_date}{sex}{expiry_date}{optional}'
            return results, ErrorCodes.NO_ERROR

    def __repr__(self) -> str:
        if self.model_type == ModelType.spotting and self.scanner:
            return f'{self.scanner.__class__.__name__}(\n{self.scanner.model})'
        elif self.model_type == ModelType.detection and self.detector:
            return f'{self.detector.__class__.__name__}(\n{self.detector.model})'
        elif self.model_type == ModelType.recognition and self.recognizer:
            return f'{self.recognizer.__class__.__name__}(\n{self.recognizer.model})'
        elif self.model_type == ModelType.two_stage and self.detector and self.recognizer:
            return (
                f'{self.detector.__class__.__name__}(\n{self.detector.model}) \n\n'
                f'{self.recognizer.__class__.__name__}(\n{self.recognizer.model})'
            )
        return 'MRZScanner(Uninitialized or Invalid Model)'

    def __call__(
        self,
        img: np.ndarray,
        do_center_crop: bool = False,
        do_postprocess: bool = False
    ) -> List[str]:
        """ Run MRZScanner.

        Args:
            img (np.ndarray): Image.
            do_center_crop (bool): Center crop.
            do_postprocess (bool): Postprocess.

        Returns:
            List[str]: List of MRZ strings.

        Raises:
            ErrorCodes: If invalid input format.
        """
        if not cb.is_numpy_img(img):
            return {
                'mrz_polygon': [],
                'mrz_texts': [''],
                'msg': ErrorCodes.INVALID_INPUT_FORMAT
            }

        if do_center_crop:
            ori_h, ori_w = img.shape[:2]
            img = cb.centercrop(img)
            new_h, new_w = img.shape[:2]
            shift = ((ori_w - new_w)//2, (ori_h - new_h)//2)

        mrz_polygon, mrz_texts = None, None
        if self.model_type == ModelType.spotting:
            mrz_texts = self.scanner(img=img)
        elif self.model_type == ModelType.detection:
            mrz_polygon = self.detector(img=img)
        elif self.model_type == ModelType.recognition:
            mrz_texts = self.recognizer(img=img)
        elif self.model_type == ModelType.two_stage:
            mrz_polygon = self.detector(img=img)
            warp_img = cb.imwarp_quadrangle(img, mrz_polygon)
            mrz_texts = self.recognizer(img=warp_img)

        if do_center_crop:
            mrz_polygon += shift

        msg = ErrorCodes.NO_ERROR
        if do_postprocess and self.model_type != ModelType.detection:
            mrz_texts, msg = self.postprocess(mrz_texts)

        return {
            'mrz_polygon': mrz_polygon,
            'mrz_texts': mrz_texts,
            'msg': msg
        }
