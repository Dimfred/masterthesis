from albumentations.core.composition import set_always_apply
import cv2 as cv
import numpy as np
import albumentations as A
import imgaug
from numpy.lib.function_base import append

import utils
from utils.augment import YoloAugmentator

from config import config


np.random.seed(1337)
imgaug.random.seed(1337)

# does not work for BBOX
# TODO for segmentation
# A.ElasticTransform(
#     always_apply=True
# )
# A.IAAPerspective(p=1),

# TODO maybe
# A.Cutout
# A.Downscale
# A.Emboss(p=1),
# A.ShiftScaleRotate()
# A.Perspective(p=1)
# A.GridDropout(p=1)


# fmt:off
augmentations = A.Compose([
    # # TODO tune params
    # # TODO bboxed still disappearing
    # # has to happen before the crop if not can happen that bboxes disappear
    # A.Rotate(
    #     limit=10,
    #     border_mode=cv.BORDER_REFLECT_101,
    #     p=0.3,
    # ),
    # A.RandomScale(scale_limit=0.1, p=0.3),
    # # THIS DOES NOT RESIZE ANYMORE THE RESIZING WAS COMMENTED OUT
    # A.RandomSizedBBoxSafeCrop(
    #     width=None, # unused
    #     height=None, # unused
    #     p=0.3,
    # ),
    # A.OneOf([
    #     A.CLAHE(p=1),
    #     A.ColorJitter(p=1),
    # ], p=0.3),
    # A.Blur(blur_limit=3, p=0.3),
    # A.GaussNoise(p=0.3),
    A.PadIfNeeded(
        min_height=1000,
        min_width=1000,
        border_mode=cv.BORDER_CONSTANT,
        value=0,
        always_apply=True
    ),
    A.Resize(
        width=config.yolo.input_size,
        height=config.yolo.input_size,
        always_apply=True
    )
], bbox_params=A.BboxParams("yolo"))
# fmt:on

import time

def main():

    start = time.time()
    data = YoloAugmentator.fileloader(config.valid_out_dir)
    data = [(cv.imread(str(img_path)), utils.load_ground_truth(label_path)) for img_path, label_path in data]
    print("Dataload:", time.time() - start)


    gstart = time.time()
    for img, labels in data:
        start = time.time()
        labels = utils.A.class_to_back(labels)

        # AUGMENT
        orig = img.copy()
        augmented = augmentations(image=img, bboxes=labels)
        img, labels = augmented["image"], augmented["bboxes"]

        labels = utils.A.class_to_front(labels)
        print("Per:", time.time() - start)
        # labels = [class_to_front(bbox) for bbox in labels]
        # utils.show_bboxes(img, labels, orig=orig)

    print("Overall:", time.time() - gstart)


if __name__ == "__main__":
    main()
