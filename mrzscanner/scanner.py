from enum import Enum
from typing import List

import capybara as cb
import numpy as np

from .spotting import Inference as SpottingInference
from .utils import replace_digits, replace_letters, replace_sex

__all__ = [
    'MRZScanner', 'ModelType', 'SpottingInference', 'ErrorCodes']


class ModelType(cb.EnumCheckMixin, Enum):
    default = 0
    spotting = 1


class ErrorCodes(Enum):
    NO_ERROR = 'No error.'
    INVALID_INPUT_FORMAT = 'Invalid input format.'
    POSTPROCESS_FAILED_LINE_COUNT = 'Postprocess failed, number of lines not 2 or 3.'
    POSTPROCESS_FAILED_TD1_LENGTH = 'Postprocess failed, length of lines not 30 when `doc_type` is TD1.'
    POSTPROCESS_FAILED_TD2_TD3_LENGTH = 'Postprocess failed, length of lines not 36 or 44 when `doc_type` is TD2 or TD3.'


class MRZScanner:

    def __init__(
        self,
        model_type: ModelType = ModelType.spotting,
        model_cfg: str = None,
        backend: cb.Backend = cb.Backend.cpu,
        gpu_id: int = 0,
        **kwargs
    ) -> None:
        """ Initialize MRZScanner.

        Args:
            model_type (ModelType): Model type.
            model_cfg (str): Model configuration.
            backend (cb.Backend): Backend.
            gpu_id (int): GPU Icb.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If invalid model_cfg is provided.
        """

        self.scanner = None
        model_type = ModelType.obj_to_enum(model_type)
        if model_type == ModelType.spotting or model_type == ModelType.default:
            model_cfg = '20240919' if model_cfg is None else model_cfg
            valid_model_cfgs = list(SpottingInference.configs.keys())
            if model_cfg not in valid_model_cfgs:
                raise ValueError(
                    f'Invalid model_cfg: {model_cfg}, '
                    f'valid model_cfgs: {valid_model_cfgs}'
                )
            self.scanner = SpottingInference(
                gpu_id=gpu_id,
                backend=backend,
                model_cfg=model_cfg,
                **kwargs
            )

    def list_models(self) -> List[str]:
        return list(self.scanner.configs.keys())

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
            results[0] = f'{doc}{country}{doc_number}{
                doc_number_hash}{optional}'
            # Line2
            birth_date = replace_letters(results[1][0:7])
            sex = replace_sex(results[1][7])
            expiry_date = replace_letters(results[1][8:15])
            nationality = replace_digits(results[1][15:18])
            optional = results[1][18:30]
            results[1] = f'{birth_date}{sex}{
                expiry_date}{nationality}{optional}'
            return results, ErrorCodes.NO_ERROR

        elif doc_type == 2:  # TD2 or TD3
            if (len(results[0]) != 36 or len(results[1]) != 36) \
                    and (len(results[0]) != 44 or len(results[1]) != 44):
                return [''], ErrorCodes.POSTPROCESS_FAILED_TD2_TD3_LENGTH
            # Line2
            doc_number = results[1][0:9]
            doc_number_hash = replace_letters(results[1][9])
            nationality = replace_digits(results[1][10:13])
            birth_date = replace_letters(results[1][13:20])
            sex = replace_sex(results[1][20])
            expiry_date = replace_letters(results[1][21:28])
            optional = results[1][28:]
            results[1] = f'{doc_number}{doc_number_hash}{
                nationality}{birth_date}{sex}{expiry_date}{optional}'
            return results, ErrorCodes.NO_ERROR

    def __call__(
        self,
        img: np.ndarray,
        do_center_crop: bool = False,
        do_postprocess: bool = True
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
            return [''], ErrorCodes.INVALID_INPUT_FORMAT
        result = self.scanner(img=img, do_center_crop=do_center_crop)

        msg = ErrorCodes.NO_ERROR
        if do_postprocess:
            result, msg = self.postprocess(result)
        return result, msg

    def __repr__(self) -> str:
        return f'{self.scanner.__class__.__name__}({self.scanner.model})'
