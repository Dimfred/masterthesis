#!/usr/bin/env python3.8

import os
import argparse
import time
from pathlib import Path
import cv2 as cv
import sys

from typing import List

parser = argparse.ArgumentParser(
    description="Rotates and flips every image in a folder and changes yolo labels and bbox's according to a transition state"
)

# parser.add_argument(
#     "-p", "--label_dir", type=str, help="the path where the files are appearing"
# )

# parser.add_argument("-t", "--train_prefix", type=str, help="path prefix for train.txt")

parser.add_argument(
    "-d",
    "--delete",
    type=bool,
    required=False,
    help="deletes all augmented files in this dir",
)

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


def parse_classes(label_dir: Path):
    with open(str(label_dir / "classes.txt"), "r") as f:
        return [f.replace("\n", "") for f in f.readlines()]


def parse_yolo(yolo_file: Path):
    with open(str(yolo_file), "r") as f:
        lines = f.readlines()
        content = []
        for line in lines:
            l, x, y, w, h = line.split(" ")
            content.append((int(l), float(x), float(y), float(w), float(h)))

        return content


def get_imgs_to_augment(label_dir: Path):
    files_to_augment = os.listdir(str(label_dir))
    files_to_augment.pop(files_to_augment.index("classes.txt"))
    # exclude already augmented files
    files_to_augment = [f for f in files_to_augment if not f.startswith("aug")]
    # exclude yolo files
    files_to_augment = [f for f in files_to_augment if not f.endswith(".txt")]
    return files_to_augment


class YoloAugmentator:
    def __init__(
        self,
        label_dir: Path,
        classes: List[str],
        rot_transition: dict,
        flip_transition: dict,
    ):
        self.label_dir = label_dir
        self.classes = classes
        self.rot_transition = rot_transition
        self.flip_transition = flip_transition

    def augment(self, file: str, oimg, ocontent):
        img90 = augmentator.rotate(oimg)
        content90 = augmentator.calc_rotations(ocontent)
        augmentator.write(file, img90, content90, 90)

        img180 = augmentator.rotate(img90)
        content180 = augmentator.calc_rotations(content90)
        augmentator.write(file, img180, content180, 180)

        img270 = augmentator.rotate(img180)
        content270 = augmentator.calc_rotations(content180)
        augmentator.write(file, img270, content270, 270)

        # flip
        fimg = augmentator.flip(oimg)
        fcontent = augmentator.calc_flips(ocontent)
        augmentator.write(file, fimg, fcontent, 0, True)

        fimg90 = augmentator.rotate(fimg)
        fcontent90 = augmentator.calc_rotations(fcontent)
        augmentator.write(file, fimg90, fcontent90, 90, True)

        fimg180 = augmentator.rotate(fimg90)
        fcontent180 = augmentator.calc_rotations(fcontent90)
        augmentator.write(file, fimg180, fcontent180, 180, True)

        fimg270 = augmentator.rotate(fimg180)
        fcontent270 = augmentator.calc_rotations(fcontent180)
        augmentator.write(file, fimg270, fcontent270, 270, True)

    def write(
        self,
        filename: str,
        img,
        content: list,
        degree: int,
        flip: bool = False,
    ):
        img_filename = "aug_" + ("flip_" if flip else "") + str(degree) + "_" + filename
        yolo_filename = img_filename.split(".")[0] + ".txt"

        cv.imwrite(str(self.label_dir / img_filename), img)

        with open(str(self.label_dir / yolo_filename), "w") as yolo_file:
            for c in content:
                yolo_file.write(" ".join(str(i) for i in c))
                yolo_file.write("\n")

            print("Augmented: ", yolo_filename)

    def create_train(self, train_prefix: str):
        yolo_files = self._get_yolo_files()
        imgs = [(f.split(".")[0] + ".jpg") for f in yolo_files]
        with open(str(self.label_dir / "train_yolo.txt"), "w") as f:
            for img in imgs:
                f.write(str(Path(train_prefix) / img) + "\n")

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

    def _get_yolo_files(self):
        return [
            f
            for f in os.listdir(self.label_dir)
            if (
                f.endswith(".txt")
                and (f != "classes.txt")
                and (f != "train.txt")
                and (f != "train_yolo.txt")
                and (f != "valid.txt")
            )
        ]

    def summary(self):
        # just some summary how much of each class we have
        summary = {}
        yolo_files = self._get_yolo_files()
        for f in yolo_files:
            with open(str(label_dir / f), "r") as yolo_file:
                lines = yolo_file.readlines()
                for line in lines:
                    label = int(line.split(" ")[0])
                    name = classes[label]
                    if name not in summary:
                        summary[name] = 0

                    summary[name] += 1

        summary = sorted(list(summary.items()), key=lambda x: x[0])
        print("")
        print("-" * 60)
        print("Class summary")
        for name, num in summary:
            print(f"{name}: {num}")


if __name__ == "__main__":
    args = parser.parse_args()

    #label_dir = Path(args.label_dir)
    label_dir = Path("data")
    files = os.listdir(str(label_dir))

    if args.delete:
        for f in files:
            if f.startswith("aug_"):
                f = str(label_dir / f)
                os.remove(f)
                print("Remove: ", f)
        os.remove(str(label_dir / "train_yolo.txt"))

        sys.exit()

    classes = parse_classes(label_dir)

    imgs_to_augment = get_imgs_to_augment(label_dir)

    augmentator = YoloAugmentator(
        label_dir, classes, label_transition_rotation, label_transition_flip
    )

    for file in imgs_to_augment:
        yolo_file_name = label_dir / (file.split(".")[0] + ".txt")

        try:
            # get original data
            ocontent = parse_yolo(yolo_file_name)
            oimg = cv.imread(str(label_dir / file))
        except Exception as e:  # file does not exists hence we haven't labeled it yet
            # print(e)
            continue

        # create 3 rotations of the original img
        # + flip and 3 rotations of the flipped img
        # => 7 more imgs per original_img
        augmentator.augment(file, oimg, ocontent)

    augmentator.summary()
    augmentator.create_train("train_yolo")
