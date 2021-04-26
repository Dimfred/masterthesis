import numpy as np
import cv2 as cv

from config import config
import utils


def rem_rot_all(dir_):
    paths = utils.list_imgs(dir_)

    for path in paths:
        print(path)
        img = cv.imread(str(path))
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

        cv.imwrite(str(path), img)

def main():
    rem_rot_all(config.train_dir)
    rem_rot_all(config.valid_dir)
    rem_rot_all(config.test_dir)

if __name__ == "__main__":
    main()
