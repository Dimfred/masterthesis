import albumentations as A
import cv2 as cv
import numpy as np

import utils
from config import config

# fmt:off
sharpen = A.Compose([
    A.Sharpen(always_apply=True)
])
# fmt:on

bg_paths = utils.list_imgs(config.backgrounds_dir)
for bg_path in bg_paths:
    img = cv.imread(str(bg_path))
    sharpened = sharpen(image=img)["image"]

    utils.show(img, sharpened, superpix)
