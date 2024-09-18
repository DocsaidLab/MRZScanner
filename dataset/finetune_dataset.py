from pathlib import Path
from typing import Tuple, Union

import albumentations as A
import cv2
import docsaidkit as D
import numpy as np

DIR = D.get_curdir(__file__)


class MRZFinetuneDataset:

    def __init__(
        self,
        root: Union[str, Path] = '/data/Dataset/MIDV2020',
        image_size: Tuple[int, int] = None,
        aug_ratio: float = 0.0,
        return_tensor: bool = False,
        **kwargs
    ):
        self.image_size = image_size
        self.root = Path(root) / 'dataset'
        self.return_tensor = return_tensor

        ds_midv2020 = []
        gt_codebook = {
            'aze_passport': {},
            'grc_passport': {},
            'lva_passport': {},
            'srb_passport': {}
        }

        for target, val in gt_codebook.items():
            data = D.load_json(
                self.root / f'templates/annotations/{target}.json')
            for d in data['_via_img_metadata'].values():
                for region in d['regions']:
                    if region['region_attributes']['field_name'] == 'mrz_line0':
                        mrz1 = region['region_attributes']['value']
                    if region['region_attributes']['field_name'] == 'mrz_line1':
                        mrz2 = region['region_attributes']['value']
                val[d['filename'].replace('.jpg', '')] = mrz1 + '&' + mrz2

        for f in D.Tqdm(D.get_files(self.root / 'images/', suffix=['.jpg'])):
            if 'aze_passport' in str(f):
                target = 'aze_passport'
            elif 'grc_passport' in str(f):
                target = 'grc_passport'
            elif 'lva_passport' in str(f):
                target = 'lva_passport'
            elif 'srb_passport' in str(f):
                target = 'srb_passport'
            else:
                continue
            key = f.stem if len(f.stem) == 2 else f.parent.stem
            gt = gt_codebook[target][key]
            ds_midv2020.append((f, gt))

        self.ds_midv2020 = ds_midv2020
        self.aug_midv = A.Compose(
            transforms=[
                A.ColorJitter(),
                A.SafeRotate(
                    limit=20,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1
                ),
                A.OneOf([
                    A.Emboss(),
                    A.Equalize(),
                ]),
                A.OneOf([
                    A.ToGray(),
                    A.InvertImg(),
                    A.ChannelShuffle(),
                ]),
            ],
            p=aug_ratio
        )

    def to_tensor(self, img: np.ndarray) -> np.ndarray:
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.
        return img

    def __len__(self):
        return len(self.ds_midv2020)

    def __getitem__(self, idx):
        img_path, gt = self.ds_midv2020[idx]
        img = D.imread(img_path)
        img = D.imresize(img, size=self.image_size)
        img = self.aug_midv(image=img)['image']

        if self.return_tensor:
            img = self.to_tensor(img)

        return img, gt
