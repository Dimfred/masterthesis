#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
import cv2 as cv
import numpy as np

from pathlib import Path
from typing import Union
import os
import shutil as sh

import utils
from config import config


class BackgroundMerger:
    def __init__(
        self,
        img,
        fg_img,
        bg_img,
    ):
        self.fg_img = img
        self.fg_mask = fg_img
        self.bg_img = bg_img
        self.fit_bg()

        self.merged = None

    def merge(self, debug: bool = False):
        # if debug:
        #     print("DEBUG: bg img")
        #     utils.show(self.bg_img)
        #     print("DEBUG: fg img")
        #     utils.show(self.fg_img)
        #     print("DEBUG: fg mask")
        #     utils.show(self.fg_mask)

        self.merged = self.bg_img.copy()

        self.merged *= np.logical_not(self.fg_mask)
        # if debug:
        #     print("DEBUG: removed fg from bg")
        #     utils.show(self.merged)

        self.merged += self.fg_img * self.fg_mask
        if debug:
            # print("DEBUG: fg with bg")
            utils.show(self.merged)

        return self.merged

    def max_axis(self, img):
        h, w = img.shape[:2]
        return "y" if h > w else "x"

    def fit_bg(self):
        fg_max = self.max_axis(self.fg_img)
        bg_max = self.max_axis(self.bg_img)

        # fit the max axis together
        if fg_max != bg_max:
            self.bg_img = cv.rotate(self.bg_img, cv.ROTATE_90_CLOCKWISE)

        fgh, fgw = self.fg_img.shape[:2]
        bgh, bgw = self.bg_img.shape[:2]

        if bgh < fgh:
            # print("DEBUG: zero pad height")
            # zero pad height
            missing_height = fgh - bgh
            self.bg_img = np.pad(
                self.bg_img, ((0, missing_height), (0, 0), (0, 0)), mode="constant"
            )

        elif bgw < fgw:
            # print("DEBUG: zero pad width")
            # zero pad width
            missing_width = fgw - bgw
            self.bg_img = np.pad(
                self.bg_img, ((0, 0), (0, missing_width), (0, 0)), mode="constant"
            )

        else:
            # crop the background
            self.bg_img = self.bg_img[:fgh, :fgw]


if __name__ == "__main__":
    # clear before applying anything
    dir_list = config.merged_dir.glob("**/*.*")
    for f in dir_list:
        os.remove(str(f))


    bg_paths = config.backgrounds_dir.glob("**/*.*")
    fg_paths = list(config.foregrounds_dir.glob("**/*.*"))
    img_paths = [utils.img_from_fg(config.train_dir, fg_path) for fg_path in fg_paths]

    def read_and_store(path, target):
        img = cv.imread(str(path))
        img = utils.resize_max_axis(img, 1000)
        target.append((path, img))

    bgs, fgs, imgs = [], [], []
    with ThreadPoolExecutor(max_workers=32) as executor:
        for path in bg_paths:
            executor.submit(read_and_store, path, bgs)

        for path in fg_paths:
            executor.submit(read_and_store, path, fgs)

        for path in img_paths:
            executor.submit(read_and_store, path, imgs)

    imgs = dict(imgs)
    for i in range(len(fgs)):
        fgs[i][1][fgs[i][1] == 2] = 0
        fgs[i][1][fgs[i][1] == 3] = 1

    for bg_path, bg_img in bgs:
        print("Projecting on: ", bg_path)
        for fg_path, fg_img in fgs:
            print("\tFG:", fg_path)

            img_path = utils.img_from_fg(config.train_dir, fg_path)
            if not img_path.exists():
                valid_img_path = utils.img_from_fg(config.valid_dir, fg_path)
                test_img_path = utils.img_from_fg(config.test_dir, fg_path)
                if not (valid_img_path.exists() or test_img_path.exists()):
                    print("-------------------------------------------------------")
                    print("-------------------------------------------------------")
                    print("-------------------------------------------------------")
                    print("IMG_NAME NOT FOUND !!!!!!!!!!!!!!!!!! SHOULD NOT HAPPEN")
                    print("-------------------------------------------------------")
                    print("-------------------------------------------------------")
                    print("-------------------------------------------------------")

                continue

            img = imgs[img_path]

            merger = BackgroundMerger(img, fg_img, bg_img)
            merged_img = merger.merge(debug=False)

            # save the merged img
            merged_path = config.merged_dir / utils.merged_name(img_path, bg_path)
            cv.imwrite(str(merged_path), merged_img)

            # copy the corresponding yolo label to the merged_dir
            yolo_label_path = utils.Yolo.label_from_img(img_path)
            merged_yolo_label_path = utils.Yolo.label_from_img(merged_path)
            sh.copy(yolo_label_path, merged_yolo_label_path)

            # copy the corresponding segmentation label to the merged_dir
            seg_label_path = utils.segmentation_label_from_img(img_path)
            merged_seg_label_path = utils.segmentation_label_from_img(merged_path)
            # only if the seg label exists copy it
            if seg_label_path.exists():
                sh.copy(seg_label_path, merged_seg_label_path)
