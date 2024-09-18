import docsaidkit as D
from finetune_dataset import MRZFinetuneDataset

ds = MRZFinetuneDataset(
    image_size=(512, 512),
    aug_ratio=0,
)

img, gt = ds[0]

D.imwrite(img, 'img.jpg')
print('GT:', gt.split('&'))
