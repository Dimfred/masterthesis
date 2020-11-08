#!/usr/bin/env python3.8

import os
import argparse
import time
from pathlib import Path
import cv2 as cv
import sys
import os
import shutil
from tabulate import tabulate

from typing import List


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
        img90 = self.rotate(oimg)
        content90 = self.calc_rotations(ocontent)
        self.write(file, img90, content90, 90)

        img180 = self.rotate(img90)
        content180 = self.calc_rotations(content90)
        self.write(file, img180, content180, 180)

        img270 = self.rotate(img180)
        content270 = self.calc_rotations(content180)
        self.write(file, img270, content270, 270)

        # flip
        fimg = self.flip(oimg)
        fcontent = self.calc_flips(ocontent)
        self.write(file, fimg, fcontent, 0, True)

        fimg90 = self.rotate(fimg)
        fcontent90 = self.calc_rotations(fcontent)
        self.write(file, fimg90, fcontent90, 90, True)

        fimg180 = self.rotate(fimg90)
        fcontent180 = self.calc_rotations(fcontent90)
        self.write(file, fimg180, fcontent180, 180, True)

        fimg270 = self.rotate(fimg180)
        fcontent270 = self.calc_rotations(fcontent180)
        self.write(file, fimg270, fcontent270, 270, True)

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
        imgs = [(f.split(".")[0] + ".jpg") for f in yolo_files]

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
        summary = {}

        label_files = self._get_label_files()
        for f in label_files:
            with open(str(self.preprocessed_dir / f), "r") as yolo_file:
                lines = yolo_file.readlines()
                for line in lines:
                    label = int(line.split(" ")[0])
                    name = self.classes[label]
                    if name not in summary:
                        summary[name] = 0

                    summary[name] += 1

        summary = sorted(list(summary.items()), key=lambda x: x[0])
        summary = [("Labels", "Number")] + summary
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
            self.make_symlink(img_filename)

            # create 3 rotations of the original img + flip and 3 rotations of the
            # flipped img => 7 more imgs per original_img
            original_labels = self._parse_labels(label_filepath)
            original_image = cv.imread(str(label_dir / img_filename))
            self.augment(img_filename, original_image, original_labels)

        self.summary()
        self.create_train()


# transition occurs always with the clock (90°)
label_transition_rotation = {
    "diode_left": "diode_top",
    "diode_top": "diode_right",
    "diode_right": "diode_bot",
    "diode_bot": "diode_left",
    "battery_left": "battery_top",
    "battery_top": "battery_right",
    "battery_right": "battery_bot",
    "battery_bot": "battery_left",
    "resistor_de_hor": "resistor_de_ver",
    "resistor_de_ver": "resistor_de_hor",
    "resistor_usa_hor": "resistor_usa_ver",
    "resistor_usa_ver": "resistor_usa_hor",
    "capacitor_hor": "capacitor_ver",
    "capacitor_ver": "capacitor_hor",
    "ground_left": "ground_top",
    "ground_top": "ground_right",
    "ground_right": "ground_bot",
    "ground_bot": "ground_left",
    "lamp_de_hor": "lamp_de_ver",
    "lamp_de_ver": "lamp_de_hor",
    "lamp_usa_hor": "lamp_usa_ver",
    "lamp_usa_ver": "lamp_usa_hor",
    "inductor_de_hor": "inductor_de_ver",
    "inductor_de_ver": "inductor_de_hor",
    "inductor_usa_hor": "inductor_usa_ver",
    "inductor_usa_ver": "inductor_usa_hor",
}

label_transition_flip = {
    "diode_left": "diode_right",
    "diode_top": "diode_top",
    "diode_right": "diode_left",
    "diode_bot": "diode_bot",
    "battery_left": "battery_right",
    "battery_top": "battery_top",
    "battery_right": "battery_left",
    "battery_bot": "battery_bot",
    "resistor_de_hor": "resistor_de_hor",
    "resistor_de_ver": "resistor_de_ver",
    "resistor_usa_hor": "resistor_usa_hor",
    "resistor_usa_ver": "resistor_usa_ver",
    "capacitor_hor": "capacitor_hor",
    "capacitor_ver": "capacitor_ver",
    "ground_left": "ground_right",
    "ground_top": "ground_top",
    "ground_right": "ground_left",
    "ground_bot": "ground_bot",
    "lamp_de_hor": "lamp_de_hor",
    "lamp_de_ver": "lamp_de_ver",
    "lamp_usa_hor": "lamp_usa_hor",
    "lamp_usa_ver": "lamp_usa_ver",
    "inductor_de_hor": "inductor_de_hor",
    "inductor_de_ver": "inductor_de_ver",
    "inductor_usa_hor": "inductor_usa_hor",
    "inductor_usa_ver": "inductor_usa_ver",
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
    label_dir = Path("data/labeled")
    preprocessed_dir = Path("data/preprocessed")

    augmentator = YoloAugmentator(
        label_dir,
        preprocessed_dir,
        label_dir_files_to_ignore,
        label_transition_rotation,
        label_transition_flip,
    )

    augmentator.run()
