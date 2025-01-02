from typing import List

import capybara as cb
import cv2
import numpy as np

from ..utils import DecodeMode, TextDecoder

DIR = cb.get_curdir(__file__)

__all__ = ['Inference']


class Inference:

    configs = {
        '20240919': {
            'model_path': 'mobilenetv4_conv_small_bifpn1_l6_d256_p12345_finetune_20240919_fp32.onnx',
            'file_id': '1WVFHyyjhbBHttY_fIaSO_xHG97tL6m5c',
            'img_size_infer': (512, 512),
        },
    }

    def __init__(
        self,
        gpu_id: int = 0,
        backend: cb.Backend = cb.Backend.cpu,
        model_cfg: str = '20240919',
        **kwargs
    ) -> None:
        self.root = DIR / 'ckpt'
        self.model_cfg = model_cfg
        self.cfg = cfg = self.configs[model_cfg]
        self.image_size = cfg['img_size_infer']
        model_path = self.root / cfg['model_path']
        if not cb.Path(model_path).exists():
            cb.download_from_google(
                cfg['file_id'], model_path.name, str(DIR / 'ckpt'))

        self.model = cb.ONNXEngine(model_path, gpu_id, backend, **kwargs)

        # Text en/de-coding
        keys = ["<PAD>", "<EOS>"] + \
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<&")
        chars_dict = {
            k: i
            for i, k in enumerate(keys)
        }

        self.text_dec = TextDecoder(
            chars_dict=chars_dict,
            decode_mode=DecodeMode.Normal
        )

    def preprocess(self, img: np.ndarray, do_center_crop: bool) -> np.ndarray:
        if do_center_crop:
            img = cb.centercrop(img)

        # Padding
        if img.shape[0] < img.shape[1]:  # H < W
            pad = (img.shape[1] - img.shape[0]) // 2
            padding = (pad, pad, 0, 0)
            img = cv2.copyMakeBorder(
                img, *padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            pad = (img.shape[0] - img.shape[1]) // 2
            padding = (0, 0, pad, pad)
            img = cv2.copyMakeBorder(
                img, *padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        tensor = cb.imresize(img, size=tuple(self.image_size))
        tensor = np.transpose(tensor, axes=(2, 0, 1)).astype('float32')

        # Normalize depanding on the model
        tensor = tensor / 255.0

        return tensor

    def engine(self, tensor: np.ndarray) -> np.ndarray:
        result = self.model(img=tensor[None])['text']
        return result.argmax(-1)

    def __call__(
        self,
        img: np.ndarray,
        do_center_crop: bool = False
    ) -> List[str]:
        data = self.preprocess(img, do_center_crop=do_center_crop)
        result = self.engine(data)
        result = self.text_dec(result)[0]
        result = result.split('&')
        return result
