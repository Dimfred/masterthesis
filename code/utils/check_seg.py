import cv2 as cv
import numpy as np

import utils
from config import config


# train_imgs = utils.list_imgs(config.train_dir)
# valid_imgs = utils.list_imgs(config.valid_dir)
# test_imgs = utils.list_imgs(config.test_dir)

train_imgs = utils.list_imgs(config.train_out_dir)
valid_imgs = utils.list_imgs(config.valid_out_dir)
test_imgs = utils.list_imgs(config.test_out_dir)

def look(img_paths):
    for path in img_paths:
        label_path = utils.segmentation_label_from_img(path)
        labels = np.load(label_path)
        img = cv.imread(str(path))

        where_one = labels == 1
        where_zero = labels == 0

        where_both = np.logical_or(where_one, where_zero)
        if not np.all(where_both):
            print(str(path))
            utils.show(img, labels)

look(train_imgs)
look(valid_imgs)
look(test_imgs)
