import capybara as cb

from mrzscanner import ModelType, MRZScanner

DIR = cb.get_curdir(__file__)

fs = cb.get_files('MRZScannerDev/data', suffix=['.jpg', '.png'])


model = MRZScanner(model_type=ModelType.two_stage)


if not (fp := DIR / 'output_labelme').is_dir():
    fp.mkdir(parents=True)

for f in cb.Tqdm(fs):
    img = cb.imread(f)
    result = model(img)

    infos = {
        "version": "5.4.0",
        "flags": {},
        "shapes": [
            {
                "label": "&".join(result['mrz_texts']),
                "points": result['mrz_polygon'].tolist(),
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
        ],
        "imagePath": f.name,
        "imageData": None,
        "imageHeight": img.shape[0],
        "imageWidth": img.shape[1],
    }

    cb.imwrite(img, fp / f.name)
    cb.dump_json(infos, fp / f"{f.stem}.json")
