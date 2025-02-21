from typing import Any, Callable, List, Tuple

import capybara as cb
import numpy as np

from ..utils import DecodeMode, TextDecoder

DIR = cb.get_curdir(__file__)

__all__ = ['Inference']


class Inference:

    configs = {
        '20250221': {
            'model_path': 'mrz_recognition_20250221_fp32.onnx',
            'file_id': '16t-kYHoBnI72MWDMCQ0K8GEyb90Tx2Rp',
            'img_size_infer': (64, 640),
        },
    }

    def __init__(
        self,
        gpu_id: int = 0,
        backend: cb.Backend = cb.Backend.cpu,
        model_cfg: str = '20250221',
        delimeter: str = '<SEP>',
        **kwargs
    ):
        self.root = DIR / 'ckpt'
        self.model_cfg = model_cfg
        self.cfg = cfg = self.configs[model_cfg]
        self.image_size = cfg['img_size_infer']
        self.delimeter = delimeter
        model_path = self.root / cfg['model_path']
        if not cb.Path(model_path).exists():
            cb.download_from_google(
                cfg['file_id'], model_path.name, str(DIR / 'ckpt'))

        self.model = cb.ONNXEngine(model_path, gpu_id, backend, **kwargs)
        self.input_key = list(self.model.input_infos.keys())[0]
        self.output_key = list(self.model.output_infos.keys())[0]

        keys = ["<PAD>", "<EOS>", delimeter] + \
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
        chars_dict = {
            k: i
            for i, k in enumerate(keys)
        }

        self.text_dec = TextDecoder(
            chars_dict=chars_dict,
            decode_mode=DecodeMode.Normal
        )

    def preprocess(self, img: np.ndarray, normalize: bool = False):
        tensor = cb.imresize(img, size=tuple(self.image_size))
        tensor = np.transpose(tensor, axes=(2, 0, 1)).astype('float32')
        tensor = tensor[None] / 255.0 if normalize else tensor[None]
        return {self.input_key: tensor}

    def postprocess(self, pred) -> Callable[[Tuple[Any, ...]], Any]:
        pred = pred[self.output_key].argmax(-1)
        result = self.text_dec(pred)[0]
        return result

    def __call__(self, img: np.ndarray, normalize: bool = True) -> List[str]:
        tensors = self.preprocess(img, normalize=normalize)
        preds = self.model(**tensors)
        result = self.postprocess(preds)
        result = result.split(self.delimeter)
        return result
