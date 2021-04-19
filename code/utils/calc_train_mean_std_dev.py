import numpy as np
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
import numba as nb
from tabulate import tabulate

import utils
from config import config


def main():
    imgs = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        img_paths = utils.list_imgs(config.train_out_dir)
        # img_paths = utils.list_imgs(config.train_dir)

        def read_and_store(path):
            img = cv.imread(str(path))
            imgs.append(img)

        for path in img_paths:
            executor.submit(read_and_store, path)

    res = [["Result"]]

    total_pixels = sum(np.product(img.shape) for img in imgs)
    res += [["NumTotal", total_pixels]]

    sum_pixels = sum((img.sum() for img in imgs))
    mean_pixel = sum_pixels / total_pixels / 255
    res += [["Mean", mean_pixel]]

    var_pixels = sum((((np.int16(img) / 255) - mean_pixel)**2).sum() for img in imgs)
    var_pixels = np.sqrt(var_pixels / total_pixels)
    res += [["Std", var_pixels]]

    print(tabulate(res))




if __name__ == "__main__":
    main()
