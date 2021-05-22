import albumentations as A
import cv2 as cv
import numpy as np

import utils
from config import config

# fmt:off
alpha = 0.4
lightness = 0.8
sharpen = A.Compose([
    A.Sharpen(
        alpha=(alpha, alpha),
        lightness=(lightness, lightness),
        always_apply=True
    )
])
clahe = A.Compose([
    A.CLAHE(
        always_apply=True
    )
])
# fmt:on

bg_paths = utils.list_imgs(config.backgrounds_dir)
for bg_path in bg_paths:
    img = cv.imread(str(bg_path))
    new_bg_path = bg_path.parent / f"{bg_path.stem}_clahe{bg_path.suffix}"
    # sharpened = sharpen(image=img)["image"]
    print(bg_path)
    clahed = clahe(image=img)["image"]

    print(new_bg_path)
    utils.show(img, clahed)
    cv.imwrite(str(new_bg_path), clahed)
