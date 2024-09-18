import cv2
import docsaidkit as D
from train_dataset import MRZDataset

ds = MRZDataset(
    image_size=(512, 512),
    aug_ratio=0,
)

img, gt, poly, fixed_points, mrz_points_hmap, mrz_region_hmap = ds[0]

D.imwrite(img, 'img.jpg')
D.imwrite(D.draw_polygon(img.copy(), poly, color=(0, 255, 0)), 'poly.jpg')
D.imwrite(mrz_points_hmap, 'mrz_points_hmap.jpg')
D.imwrite(mrz_region_hmap, 'mrz_region_hmap.jpg')

point_img = img.copy()
for p in fixed_points:
    if p[0] == 0 and p[1] == 0:
        break

    cv2.circle(point_img, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

D.imwrite(point_img, 'points.jpg')

print('GT:', gt.split('&'))
