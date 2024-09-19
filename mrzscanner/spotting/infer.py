from typing import List

import cv2
import docsaidkit as D
import numpy as np

DIR = D.get_curdir(__file__)

__all__ = ['Inference']


class Inference:

    configs = {
        '20240919': {
            'model_path': 'mobilenetv4_conv_small_bifpn1_l6_d256_p12345_finetune_20240919_fp32.onnx',
            'file_id': 'EpCRaBq6KxPaQNa',
            'img_size_infer': (512, 512),
        },
    }

    def __init__(
        self,
        gpu_id: int = 0,
        backend: D.Backend = D.Backend.cpu,
        model_cfg: str = '20240919',
        **kwargs
    ) -> None:
        self.root = DIR / 'ckpt'
        self.model_cfg = model_cfg
        self.cfg = cfg = self.configs[model_cfg]
        self.image_size = cfg['img_size_infer']
        model_path = self.root / cfg['model_path']
        if not D.Path(model_path).exists():
            D.download_from_docsaid(
                cfg['file_id'], model_path.name, str(model_path))

        self.model = D.ONNXEngine(model_path, gpu_id, backend, **kwargs)

        # Text en/de-coding
        keys = ["<PAD>", "<EOS>"] + \
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<&")
        chars_dict = {
            k: i
            for i, k in enumerate(keys)
        }

        self.text_dec = D.TextDecoder(
            chars_dict=chars_dict,
            decode_mode=D.DecodeMode.Normal
        )

    def preprocess(self, img: np.ndarray, do_center_crop: bool) -> np.ndarray:
        if do_center_crop:
            img = D.centercrop(img)

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

        tensor = D.imresize(img, size=tuple(self.image_size))
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
