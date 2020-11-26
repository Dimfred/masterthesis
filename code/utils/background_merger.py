#!/usr/bin/env python3

import cv2 as cv
import numpy as np

from pathlib import Path
from typing import Union
import os
import shutil

import utils
from config import config


class BackgroundMerger:
    def __init__(
        self,
        fg_img_path: Union[str, Path],
        fg_mask_path: Union[str, Path],
        bg_path: Union[str, Path],
    ):
        self.fg_img_path, self.fg_mask_path, self.bg_path = (
            fg_img_path,
            fg_mask_path,
            bg_path,
        )

        self.fg_img = cv.imread(str(fg_img_path))
        self.fg_img = utils.resize_max_axis(self.fg_img, 1000)

        self.fg_mask = cv.imread(str(fg_mask_path))
        self.fg_mask = utils.resize_max_axis(self.fg_mask, 1000)
        # TODO still dunno why this is happening
        # normally the mask should only store bools, instead some values
        # are 2 and 3
        self.fg_mask[self.fg_mask == 2] = 1
        self.fg_mask[self.fg_mask == 3] = 1

        self.bg_img = cv.imread(str(bg_path))
        self.bg_img = utils.resize_max_axis(self.bg_img, 1000)
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
    bgs = os.listdir(config.backgrounds_dir)
    fg_masks = os.listdir(config.foregrounds_dir)

    for bg in bgs:
        print("Projecting on: ", bg)
        bg = config.backgrounds_dir / bg
        for fg_mask in fg_masks:
            print("\tFG:", fg_mask)

            img_name = utils.img_from_mask(config.label_dir, fg_mask)
            fg_mask = config.foregrounds_dir / fg_mask

            merger = BackgroundMerger(img_name, fg_mask, bg)
            merged_img = merger.merge(debug=False)

            merged_name = utils.merged_name(
                os.path.basename(img_name), os.path.basename(bg)
            )
            cv.imwrite(str(config.merged_dir / merged_name), merged_img)

            # copy the label file
            original_label_file = utils.label_file_from_img(img_name)
            merged_label_file = utils.label_file_from_img(merged_name)
            shutil.copy(
                original_label_file,
                config.merged_dir / merged_label_file,
            )
