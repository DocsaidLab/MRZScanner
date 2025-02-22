from typing import Tuple

import capybara as cb
import cv2
import numpy as np

DIR = cb.get_curdir(__file__)

__all__ = ['Inference']


class Inference:

    configs = {
        '20250222': {
            'model_path': 'mrz_detection_20250222_fp32.onnx',
            'file_id': '1D--D7glsse51oIdcyerZoYs88iokOmqU',
            'img_size_infer': (256, 256),
        }
    }

    def __init__(
        self,
        gpu_id: int = 0,
        backend: cb.Backend = cb.Backend.cpu,
        model_cfg: str = '20250222',
        **kwargs
    ):
        self.root = DIR / 'ckpt'
        self.model_cfg = model_cfg
        self.cfg = cfg = self.configs[model_cfg]
        self.image_size = cfg['img_size_infer']
        model_path = self.root / cfg['model_path']
        if not cb.Path(model_path).exists():
            cb.download_from_google(
                cfg['file_id'], model_path.name, str(DIR / 'ckpt'))

        self.model = cb.ONNXEngine(model_path, gpu_id, backend, **kwargs)
        self.input_key = list(self.model.input_infos.keys())[0]
        self.output_key = list(self.model.output_infos.keys())[0]

    def preprocess(self, img: np.ndarray, normalize: bool = False):

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

        tensor = cb.imresize(img, size=self.image_size)
        tensor = np.transpose(tensor, axes=(2, 0, 1)).astype('float32')
        tensor = tensor[None] / 255.0 if normalize else tensor[None]

        return {self.input_key: tensor}, (img.shape[0], img.shape[1]), (padding[2], padding[0])

    def postprocess(self, hmap: np.ndarray, img_size: Tuple[int, int], shift: Tuple[int, int]) -> np.ndarray:
        hmap = np.uint8(hmap * 255)
        hmap = cb.imresize(hmap, size=img_size)
        hmap = cb.imbinarize(hmap)
        poly = cb.Polygons.from_image(hmap)
        if len(poly) == 0:
            return np.array([], dtype=np.float32)

        poly = poly[poly.area == poly.area.max()]
        poly = poly[0].to_min_boxpoints()
        poly = np.array(poly).astype(np.float32)
        poly -= shift

        return poly

    def __call__(self, img: np.ndarray, normalize: bool = True) -> np.ndarray:
        tensor, img_size, shift = self.preprocess(
            img, normalize=normalize)
        x = self.model(**tensor)
        polygon = self.postprocess(
            hmap=x[self.output_key][0],
            img_size=img_size,
            shift=shift
        )
        return polygon
