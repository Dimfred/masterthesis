#!/usr/bin/env python3

import os
import argparse
import time
from pathlib import Path
import cv2 as cv
import sys
import os
import shutil
from tabulate import tabulate
import copy

from typing import List
from config import config


class YoloAugmentator:
    def __init__(
        self,
        label_dir: Path,
        preprocessed_dir: Path,
        label_dir_files_to_ignore: List[str],
        rot_transition: dict,
        flip_transition: dict,
    ):
        self.label_dir = label_dir
        self.preprocessed_dir = preprocessed_dir
        self.label_dir_files_to_ignore = label_dir_files_to_ignore
        self.classes = self._parse_classes(label_dir)
        self.rot_transition = rot_transition
        self.flip_transition = flip_transition

    def augment(self, file: str, oimg, ocontent):
        # rotate original image
        img = oimg.copy()
        content = copy.deepcopy(ocontent)

        # store the original_img
        self.write(file, img, content, 0)

        for degree in (90, 180, 270):
            img = self.rotate(img)
            content = self.calc_rotations(content)
            self.write(file, img, content, degree)

        # flip
        img = self.flip(oimg)
        content = self.calc_flips(ocontent)
        self.write(file, img, content, 0, True)

        # rotate flipped image
        for degree in (90, 180, 270):
            img = self.rotate(img)
            content = self.calc_rotations(content)
            self.write(file, img, content, degree, True)

    def make_symlink(self, img_filename):
        img_sym = str(self.preprocessed_dir / img_filename)
        if os.path.exists(img_sym):
            os.remove(img_sym)

        # create a symlink in the preprocessed dir of the image
        img_abs = os.path.abspath(str(self.label_dir / img_filename))
        os.symlink(img_abs, img_sym)

        label_filename = f"{(os.path.splitext(img_filename)[0])}.txt"
        label_sym = str(self.preprocessed_dir / label_filename)
        if os.path.exists(label_sym):
            os.remove(label_sym)

        # create a symlink in the preprocessed dir of the labelfile
        label_abs = os.path.abspath(str(self.label_dir / label_filename))
        os.symlink(label_abs, label_sym)

    def write(
        self,
        filename: str,
        img,
        content: list,
        degree: int,
        flip: bool = False,
    ):
        filename, ext = os.path.splitext(filename)

        # {name}_{XXX: degree}_{flip/""}_aug.ext
        filename = "{filename}_{degree:03d}_{flip}aug".format(
            filename=filename, degree=degree, flip=("hflip_" if flip else "nflip_")
        )

        img_filename = f"{filename}{ext}"
        label_filename = f"{filename}.txt"

        cv.imwrite(str(self.preprocessed_dir / img_filename), img)

        with open(str(self.preprocessed_dir / label_filename), "w") as label_file:
            for c in content:
                label_file.write(" ".join(str(i) for i in c))
                label_file.write("\n")

            print("Augmented: ", label_filename)

    def create_train(self):
        yolo_files = self._get_label_files()

        imgs = []
        for label_file in yolo_files:
            file_name = os.path.splitext(label_file)[0]
            jpg = f"{file_name}.jpg"
            if os.path.exists(self.preprocessed_dir / jpg):
                imgs.append(jpg)
            else:
                png = f"{file_name}.png"
                imgs.append(png)

        train_yolo = "train_yolo.txt"
        with open(str(self.label_dir / train_yolo), "w") as f:
            for img in imgs:
                if "aug" in img:
                    f.write(str(self.preprocessed_dir / img) + "\n")

        os.symlink(
            os.path.abspath(self.label_dir / train_yolo),
            self.preprocessed_dir / train_yolo,
        )

        valid_yolo = "valid_yolo.txt"
        with open(str(self.label_dir / valid_yolo), "w") as f:
            for img in imgs:
                if "aug" not in img:
                    f.write(str(self.preprocessed_dir / img) + "\n")

        os.symlink(
            os.path.abspath(self.label_dir / valid_yolo),
            self.preprocessed_dir / valid_yolo,
        )

    def rotate(self, img):
        return cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

    def flip(self, img):
        return cv.flip(img, +1)  # xxyy => yyxx

    def calc_flips(self, yolo_content: list):
        return [self.calc_flip(content) for content in yolo_content]

    def calc_flip(self, content):
        l, x, y, w, h = content

        current_name = self.classes[l]
        new_name = self.flip_transition[current_name]

        new_label = self.classes.index(new_name)
        new_x = 1 - x
        return new_label, new_x, y, w, h

    """
    ==> x               y <==       ny = x
    ||           =>         ||      nx = 1 - y
    y                        x      nw = h
                                    nh = w
    """

    def calc_rotations(self, yolo_content: list):
        return list(map(self.calc_rotation, yolo_content))

    def calc_rotation(
        self, content
    ):  # l: int, x: float, y: float, w: float, h: float):
        l, x, y, w, h = content

        current_name = self.classes[l]
        new_name = self.rot_transition[current_name]

        new_label = self.classes.index(new_name)
        new_x = 1 - y
        new_y = x
        new_w = h
        new_h = w
        return new_label, new_x, new_y, new_w, new_h

    def _get_label_files(self):
        return [
            f
            for f in os.listdir(self.preprocessed_dir)
            if f.endswith(".txt") and f not in self.label_dir_files_to_ignore
        ]

    def summary(self):
        # just some summary how much of each class we have
        summary_real = {}
        summary_augmented = {}

        label_files = self._get_label_files()
        for f in label_files:
            with open(str(self.preprocessed_dir / f), "r") as yolo_file:
                lines = yolo_file.readlines()
                for line in lines:
                    label = int(line.split(" ")[0])
                    name = self.classes[label]
                    if name not in summary_augmented:
                        summary_augmented[name] = 0
                        summary_real[name] = 0

                    summary_augmented[name] += 1

                    # is original
                    if "_000_nflip_" in f:
                        summary_real[name] += 1

        summary = sorted(
            [
                (name, summary_real[name], summary_augmented[name])
                for name in summary_augmented.keys()
            ],
            key=lambda x: x[0],
        )
        summary = [("Labels", "Real", "Augmented")] + summary
        print(tabulate(summary))

    def _parse_classes(self, label_dir: Path):
        with open(str(label_dir / "classes.txt"), "r") as f:
            return [f.replace("\n", "") for f in f.readlines()]

    def _parse_labels(self, yolo_file: Path):
        with open(str(yolo_file), "r") as f:
            lines = f.readlines()
            content = []
            for line in lines:
                l, x, y, w, h = line.split(" ")
                content.append((int(l), float(x), float(y), float(w), float(h)))

        return content

    def _get_imgs_to_augment(self, label_dir: Path):
        imgs_to_augment = os.listdir(str(label_dir))

        for file in self.label_dir_files_to_ignore:
            try:
                imgs_to_augment.pop(imgs_to_augment.index(file))
            except Exception as e:
                print(e)

        # exclude already augmented files
        # files_to_augment = [f for f in files_to_augment if not "aug" in f]

        # exclude label files
        imgs_to_augment = [f for f in imgs_to_augment if not f.endswith(".txt")]
        return imgs_to_augment

    def run(self):
        for filename in os.listdir(self.preprocessed_dir):
            os.remove(self.preprocessed_dir / filename)

        os.symlink(
            os.path.abspath(self.label_dir / "classes.txt"),
            self.preprocessed_dir / "classes.txt",
        )

        for img_filename in self._get_imgs_to_augment(self.label_dir):
            label_filepath = label_dir / "{}.txt".format(
                os.path.splitext(img_filename)[0]
            )

            # create symlinks for the original image and labels in the preprocessed
            # directory
            # self.make_symlink(img_filename)

            # create 3 rotations of the original img + flip and 3 rotations of the
            # flipped img => 7 more imgs per original_img
            original_labels = self._parse_labels(label_filepath)
            original_image = cv.imread(
                str(label_dir / img_filename), cv.IMREAD_GRAYSCALE
            )
            self.augment(img_filename, original_image, original_labels)

        self.summary()
        self.create_train()


# transition occurs always with the clock (90°)
label_transition_rotation = {
    "diode_left": "diode_top",
    "diode_top": "diode_right",
    "diode_right": "diode_bot",
    "diode_bot": "diode_left",
    "bat_left": "bat_top",
    "bat_top": "bat_right",
    "bat_right": "bat_bot",
    "bat_bot": "bat_left",
    "res_de_hor": "res_de_ver",
    "res_de_ver": "res_de_hor",
    "res_us_hor": "res_us_ver",
    "res_us_ver": "res_us_hor",
    "cap_hor": "cap_ver",
    "cap_ver": "cap_hor",
    "gr_left": "gr_top",
    "gr_top": "gr_right",
    "gr_right": "gr_bot",
    "gr_bot": "gr_left",
    "lamp_de_hor": "lamp_de_ver",
    "lamp_de_ver": "lamp_de_hor",
    "lamp_us_hor": "lamp_us_ver",
    "lamp_us_ver": "lamp_us_hor",
    "ind_de_hor": "ind_de_ver",
    "ind_de_ver": "ind_de_hor",
    "ind_us_hor": "ind_us_ver",
    "ind_us_ver": "ind_us_hor",
    "source_hor": "source_ver",
    "source_ver": "source_hor",
    "current_hor": "current_ver",
    "current_ver": "current_hor",
    "edge_tl": "edge_tr",
    "edge_tr": "edge_br",
    "edge_br": "edge_bl",
    "edge_bl": "edge_tl",
    "t_left": "t_top",
    "t_top": "t_right",
    "t_right": "t_bot",
    "t_bot": "t_left",
    "cross": "cross",
}

# flip over y axis
label_transition_flip = {
    "diode_left": "diode_right",
    "diode_top": "diode_top",
    "diode_right": "diode_left",
    "diode_bot": "diode_bot",
    "bat_left": "bat_right",
    "bat_top": "bat_top",
    "bat_right": "bat_left",
    "bat_bot": "bat_bot",
    "res_de_hor": "res_de_hor",
    "res_de_ver": "res_de_ver",
    "res_us_hor": "res_us_hor",
    "res_us_ver": "res_us_ver",
    "cap_hor": "cap_hor",
    "cap_ver": "cap_ver",
    "gr_left": "gr_right",
    "gr_top": "gr_top",
    "gr_right": "gr_left",
    "gr_bot": "gr_bot",
    "lamp_de_hor": "lamp_de_hor",
    "lamp_de_ver": "lamp_de_ver",
    "lamp_us_hor": "lamp_us_hor",
    "lamp_us_ver": "lamp_us_ver",
    "ind_de_hor": "ind_de_hor",
    "ind_de_ver": "ind_de_ver",
    "ind_us_hor": "ind_us_hor",
    "ind_us_ver": "ind_us_ver",
    "source_hor": "source_hor",
    "source_ver": "source_ver",
    "current_hor": "current_hor",
    "current_ver": "current_ver",
    "edge_tl": "edge_tr",
    "edge_tr": "edge_tl",
    "edge_br": "edge_bl",
    "edge_bl": "edge_br",
    "t_left": "t_right",
    "t_top": "t_top",
    "t_right": "t_left",
    "t_bot": "t_bot",
    "cross": "cross",
}

label_dir_files_to_ignore = [
    "classes.names",
    "classes.txt",
    "train.txt",
    "val.txt",
    "val.txt.bak",
    "valid_yolo.txt",
    "train_yolo.txt",
    "log",
]

if __name__ == "__main__":

    if len(sys.argv) < 1:
        print("./augment.py <train/valid>")
        sys.exit()

    augment_type = sys.argv[1]
    if augment_type == "train":
        label_dir = config.label_dir
        preprocessed_dir = config.preprocessed_dir
    else:
        label_dir = config.valid_dir
        preprocessed_dir = config.valid_preprocessed_dir

    YoloAugmentator(
        label_dir,
        preprocessed_dir,
        label_dir_files_to_ignore,
        label_transition_rotation,
        label_transition_flip,
    ).run()
