from pathlib import Path
from typing import List, Tuple, Union

import albumentations as A
import cv2
import docsaidkit as D
import numpy as np
from wordcanvas import MRZGenerator

DIR = D.get_curdir(__file__)


class DetectionImageAug:

    def __init__(self, image_size: Tuple[int, int], p=0.5):
        h, w = image_size

        self.pixel_transform = A.Compose(
            transforms=[
                A.OneOf([
                    A.GaussianBlur(),
                    A.MotionBlur(),
                ]),
                A.OneOf([
                    A.ISONoise(),
                    A.GaussNoise(),
                ]),
                A.OneOf([
                    A.Emboss(),
                    A.Equalize(),
                    A.RandomSunFlare(src_radius=120),
                ]),
                A.OneOf([
                    A.ToGray(),
                    A.InvertImg(),
                    A.ChannelShuffle(),
                ]),
                A.ColorJitter(),
            ],
            p=p
        )

        self.spacial_transform = A.Compose(
            transforms=[
                A.RandomSizedBBoxSafeCrop(
                    height=h,
                    width=w,
                    p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=[-0.3, -0.1],
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.SafeRotate(
                    limit=45,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1
                ),
                A.Perspective(
                    scale=(0.05, 0.09),
                    keep_size=True,
                    fit_output=True,
                    pad_mode=cv2.BORDER_CONSTANT,
                    p=1
                ),
            ],
            p=p,
            keypoint_params=A.KeypointParams(
                format='xy',
                remove_invisible=False
            ),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            ),
            additional_targets={
                'points': 'keypoints',
            }
        )

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        points: np.ndarray,
        keypoints: np.ndarray
    ):
        image = self.pixel_transform(image=image)['image']
        outputs = self.spacial_transform(
            image=image,
            bboxes=[boxes],
            keypoints=keypoints,
            points=points,
            labels=[0]
        )

        img = outputs['image']
        box = outputs['bboxes'][0]
        pts = outputs['points']
        kps = D.order_points_clockwise(np.array(outputs['keypoints']))
        return img, box, kps, pts


class MRZDataset:

    def __init__(
        self,
        root: Union[str, Path] = '/data/Dataset',
        image_size: Tuple[int, int] = None,
        aug_ratio: float = 0.0,
        return_tensor: bool = False,
        length_of_dataset: int = 1000,
        **kwargs
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.return_tensor = return_tensor

        # 用於手動設定資料集的數量
        self.length_of_dataset = length_of_dataset

        # 使用 indoor_scene_recognition 資料集作為基底背景
        self.background = D.get_files(
            self.root / 'indoor_scene_recognition', suffix=['.jpg'])

        # 使用 DocVQA 資料集作為文字雜訊背景
        self.text_background = D.get_files(
            self.root / 'DocVQA', suffix=['.png'])

        # 調用 MRZGenerator 生成 MRZ 圖片（基於 wordcanvas 實作）
        self.mrz_generator = MRZGenerator(**kwargs)

        # 設定圖片增強函數
        self.aug_func = DetectionImageAug(image_size=image_size, p=aug_ratio)

    def to_tensor(
        self,
        img: np.ndarray, mask: np.ndarray,
        poly: np.ndarray, points: np.ndarray,
        mrz_hmap: np.ndarray
    ) -> np.ndarray:
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.
        mask = mask.astype(np.float32) / 255.
        poly = D.Polygon(poly).normalize(
            w=self.image_size[1], h=self.image_size[0]).numpy()
        points = D.Polygon(points).normalize(
            w=self.image_size[1], h=self.image_size[0]).numpy()
        mrz_hmap = mrz_hmap[..., 0].astype(np.float32) / 255.
        return img, mask, poly, points, mrz_hmap

    def gen_gaussian_point(self, mask, poly, ksize: int = 7):
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getGaussianKernel(ksize, sigma=ksize//3)
        kernel = np.outer(kernel, kernel)

        half_ksize = ksize // 2
        mask_h, mask_w = mask.shape[:2]
        mask = D.pad(mask, pad_size=(half_ksize, half_ksize))

        for p in poly:
            x, y = int(p[0]), int(p[1])
            if x <= 0 or y <= 0 or x >= mask_w or y >= mask_h:
                continue

            x += half_ksize
            y += half_ksize

            min_y = y - half_ksize
            max_y = y + half_ksize + 1
            min_x = x - half_ksize
            max_x = x + half_ksize + 1

            kernel = (kernel - kernel.min()) / \
                (kernel.max() - kernel.min()) * 255

            mask[min_y:max_y, min_x:max_x] = np.uint8(kernel)

        mask = mask[half_ksize:-half_ksize, half_ksize:-half_ksize]

        return mask

    def apply_transparency(self, img, transparency=0.5):
        height, width = img.shape[:2]
        background = np.ones((height, width, 3), dtype=np.uint8) * 255
        output_img = cv2.addWeighted(
            img, transparency, background, 1 - transparency, 0)
        return output_img

    def apply_mrz_image(
        self,
        img: np.ndarray,
        mrz_img: np.ndarray,
        mrz_points: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, D.Box, np.ndarray]:

        ori_h = mrz_img.shape[0]
        ori_w = mrz_img.shape[1]

        random_x1 = int(img.shape[1] * np.random.uniform(0.02, 0.15))
        random_x2 = int(img.shape[1] * np.random.uniform(0.85, 0.98))
        random_y1 = int(img.shape[0] * np.random.uniform(0.02, 0.8))

        mrz_img = D.imresize(
            mrz_img,
            size=(None, random_x2 - random_x1),
            interpolation=D.INTER.NEAREST
        )

        slice_y = slice(random_y1, random_y1 + mrz_img.shape[0])
        slice_x = slice(random_x1, random_x2)
        img[slice_y, slice_x] = np.where(
            mrz_img == (0, 0, 0),
            mrz_img + np.random.randint(0, 150, 3),
            img[slice_y, slice_x]
        )

        box = D.Box([
            random_x1,
            random_y1,
            random_x2,
            random_y1 + mrz_img.shape[0]
        ])

        poly = box.to_polygon().numpy()

        mrz_points = np.array(mrz_points) * [mrz_img.shape[1] / ori_w,
                                             mrz_img.shape[0] / ori_h]
        mrz_points += [random_x1, random_y1]

        return img, box, poly, mrz_points

    def apply_text(self, img):
        idx = np.random.randint(0, len(self.text_background))
        text_img = D.imread(self.text_background[idx])
        text_img = A.RandomResizedCrop(height=img.shape[0], width=img.shape[1])(
            image=text_img)['image']
        text_img = np.stack([D.imbinarize(text_img)] * 3, axis=-1)
        img = np.where(
            text_img == (0, 0, 0),
            text_img,
            img
        )
        return img

    def __len__(self):
        return self.length_of_dataset

    def __getitem__(self, idx):

        # 隨機選擇背景圖片
        idx = np.random.randint(0, len(self.background))
        img = D.imread(self.background[idx])

        # 隨機生成 MRZ 文字和圖片
        mrz_infos = self.mrz_generator()
        gt = mrz_infos['text']
        points = mrz_infos['points']
        mrz_img = mrz_infos['image']

        # 貼上文字基底雜訊
        img = self.apply_text(img)

        # 覆蓋隨機透明度
        img = self.apply_transparency(img, np.random.uniform(0, 0.6))

        # 縮放基底圖片至輸出大小
        img = D.imresize(img, size=self.image_size)

        # 貼上 MRZ 圖片
        img, box, poly, points = self.apply_mrz_image(
            img, mrz_img, points)

        # 套用圖片增強
        img, box, poly, points = self.aug_func(
            image=img,
            boxes=box,
            points=points,
            keypoints=poly
        )

        # 繪製 MRZ 文字的點熱圖
        mrz_points_hmap = np.zeros(
            (img.shape[0], img.shape[1]), dtype=np.uint8)
        mrz_points_hmap = self.gen_gaussian_point(mrz_points_hmap, points)

        # 繪製 MRZ 文字區域的二元分割圖
        mrz_region_hmap = D.draw_polygon(
            np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8),
            polygon=poly,
            color=(255, 255, 255),
            fillup=True
        )

        # 固定輸出點的數量為 90 個，不足補 0
        fixed_points = np.zeros((90, 2), dtype=np.float32)
        if len(points) > 0:
            fixed_points[:len(points)] = points

        if self.return_tensor:
            img, mrz_points_hmap, poly, fixed_points, mrz_region_hmap = \
                self.to_tensor(
                    img, mrz_points_hmap, poly, fixed_points, mrz_region_hmap)

        return img, gt, poly, fixed_points, mrz_points_hmap, mrz_region_hmap
