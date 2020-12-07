#!/usr/bin/env python3


import cv2 as cv
import numpy as np

import os
import shutil as sh
from pathlib import Path
import click
from tabulate import tabulate
import copy

from typing import List, Union, Tuple, Callable
from easydict import EasyDict

from config import config
import utils


class CircuitAugmentator:
    def __init__(
        self,
        train_dir: Path,
        train_out_dir: Path,
        valid_dir: Path,
        valid_out_dir: Path,
        merged_dir: Path,
        img_params: EasyDict,
        fileloader: Callable[[Path], List[Tuple[Path, Path]]],
        # receives path to labels, returns List[(abs_img_path, abs_label_path)]
        clean: bool = True,
    ):
        self.train_dir = train_dir
        self.train_out_dir = train_out_dir

        self.valid_dir = valid_dir
        self.valid_out_dir = valid_out_dir

        self.merged_dir = merged_dir

        self.img_params = img_params

        self.train_files = fileloader(self.train_dir)
        if self.merged_dir is not None:
            self.train_files.extend(fileloader(self.merged_dir))

            by_img_name = lambda x: x[0]
            self.train_files = sorted(self.train_files, key=by_img_name)

        self.valid_files = fileloader(self.valid_dir)

        if clean:
            self.clean(self.train_out_dir)
            self.clean(self.valid_out_dir)

    def imread(self, path: Path):
        _imread_type = (
            cv.IMREAD_COLOR if self.img_params.channels == 3 else cv.IMREAD_GRAYSCALE
        )
        img = cv.imread(str(path), _imread_type)

        # TODO else
        if self.img_params.keep_ar and self.img_params.resize:
            img = utils.resize_max_axis(img, self.img_params.resize)

        return img

    def imwrite(self, path: Path, img: np.ndarray):
        cv.imwrite(str(path), img)

    def clean(self, path: Path):
        # TODO sucks
        for filename in os.listdir(path):
            os.remove(path / filename)


class UNetAugmentator(CircuitAugmentator):
    def __init__(
        self,
        train_dir: Path,
        train_out_dir: Path,
        valid_dir: Path,
        valid_out_dir: Path,
        merged_dir: Path,
        img_params: EasyDict,
        # receives label_dir, returns List[(abs_img_path, abs_label_path)]
        clean: bool = True,
    ):
        super().__init__(
            train_dir,
            train_out_dir,
            valid_dir,
            valid_out_dir,
            merged_dir,
            img_params,
            UNetAugmentator.fileloader,
            clean,
        )

    @staticmethod
    def fileloader(path: Path):
        img_paths = utils.list_imgs(path)

        img_label_paths = []
        for img_path in img_paths:
            label_path = utils.segmentation_label_from_img(img_path)
            if label_path.exists():
                img_label_paths.append((img_path, label_path))

        return img_label_paths

    def perform(self, files: List[Tuple[Path, Path]], output_dir: Path):
        for img_path, label_path in files:
            img = self.imread(img_path)

            self.imwrite(output_dir / img_path.name, img)
            sh.copy(label_path, output_dir / label_path.name)

    def run(self):
        self.perform(self.train_files, self.train_out_dir)
        self.perform(self.valid_files, self.valid_out_dir)


class YoloAugmentator(CircuitAugmentator):
    def __init__(
        self,
        train_dir: Path,
        train_out_dir: Path,
        valid_dir: Path,
        valid_out_dir: Path,
        merged_dir: Path,
        img_params: EasyDict,
        # receives label_dir, returns List[(abs_img_path, abs_label_path)]
        rot_transition: dict,
        flip_transition: dict,
        # perform_augmentation: bool,
        clean: bool = True,
    ):
        super().__init__(
            train_dir,
            train_out_dir,
            valid_dir,
            valid_out_dir,
            merged_dir,
            img_params,
            YoloAugmentator.fileloader,
            clean,
        )

        self.rot_transition = rot_transition
        self.flip_transition = flip_transition
        # self.perform_augmentation = perform_augmentation
        self.classes = self._parse_classes(self.train_dir)

    @staticmethod
    def fileloader(path: Path):
        img_paths = utils.list_imgs(path)

        img_label_paths = []
        for img_path in img_paths:
            label_path = utils.yolo_label_from_img(img_path)
            if label_path.exists():
                img_label_paths.append((img_path, label_path))

        return img_label_paths

    def run(self):
        self.copy_classes(self.train_dir, self.train_out_dir)
        self.perform(self.train_files, self.train_out_dir, perform_augmentation=True)

        self.copy_classes(self.valid_dir, self.valid_out_dir)
        self.perform(self.valid_files, self.valid_out_dir, perform_augmentation=False)

    def perform(
        self,
        files: List[Tuple[Path, Path]],
        output_dir: Path,
        perform_augmentation: bool,
    ):
        for img_path, label_path in files:
            labels = self._parse_labels(label_path)

            img = self.imread(img_path)
            self.augment(img_path, img, labels, output_dir, perform_augmentation)

    def copy_classes(self, src_dir: Path, dst_dir: Path):
        sh.copy(src_dir / "classes.txt", dst_dir / "classes.txt")

    def augment(
        self, file: str, oimg, ocontent, output_dir: Path, perform_augmentation: bool
    ):
        # rotate original image
        img = oimg.copy()
        content = copy.deepcopy(ocontent)

        # store the original_img
        self.write(file, img, content, 0, False, output_dir)

        # normally we don't perform valid augmentation hence just cpy
        if not perform_augmentation:
            return

        for degree in (90, 180, 270):
            img = self.rotate(img)
            content = self.calc_rotations(content)
            self.write(file, img, content, degree, False, output_dir)

        # flip
        img = self.flip(oimg)
        content = self.calc_flips(ocontent)
        self.write(file, img, content, 0, True, output_dir)

        # rotate flipped image
        for degree in (90, 180, 270):
            img = self.rotate(img)
            content = self.calc_rotations(content)
            self.write(file, img, content, degree, True, output_dir)

    def write(
        self,
        img_path: Path,
        img,
        content: list,
        degree: int,
        flip: bool,
        output_dir: Path,
    ):
        # {name}_{XXX: degree}_{flip/""}_aug.ext
        filename = "{filename}_{degree:03d}_{flip}aug".format(
            filename=img_path.stem,
            degree=degree,
            flip=("hflip_" if flip else "nflip_"),
        )

        img_filename = f"{filename}{img_path.suffix}"
        label_filename = f"{filename}.txt"

        cv.imwrite(str(output_dir / img_filename), img)

        with open(str(output_dir / label_filename), "w") as label_file:
            for c in content:
                label_file.write(" ".join(str(i) for i in c))
                label_file.write("\n")

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

    def create_labels_file(self, output_dir: Path):
        img_label_paths = YoloAugmentator.fileloader(output_dir)

        label_file = "labels.txt"
        with open(str(output_dir / label_file), "w") as f:
            for img_path, _ in img_label_paths:
                print(img_path, file=f)

    def summary(self):
        # reparse classes after stripper hits
        classes = self._parse_classes(self.train_out_dir)

        real_train, aug_train = self._summary(classes, self.train_out_dir)
        real_valid, aug_valid = self._summary(classes, self.valid_out_dir)

        class_names = sorted(real_train.keys())

        summary_count = np.vstack(
            [
                (
                    real_train[class_name],
                    real_valid[class_name],
                    aug_train[class_name],
                    aug_valid[class_name],
                )
                for class_name in class_names
            ]
        )

        trains = summary_count[:, 0]
        valids = summary_count[:, 1]
        ratios = valids / trains

        reals = summary_count[:, :2].copy()
        augs = summary_count[:, 2:].copy()
        summary_count = np.append(reals, ratios[:, np.newaxis], axis=1)
        summary_count = np.append(summary_count, augs, axis=1)

        # accumulate summary idxs with common base class
        idxs_to_combine = {}
        for idx, sub_class_name in enumerate(class_names):
            class_name, *_ = sub_class_name.split("_", 1)

            if class_name not in idxs_to_combine:
                idxs_to_combine[class_name] = []

            idxs_to_combine[class_name].append(idx)

        # merge base classes
        new_class_names = sorted(idxs_to_combine.keys())
        new_rows = []
        for class_name in new_class_names:
            idxs = idxs_to_combine[class_name]
            stacked = np.vstack([summary_count[idx] for idx in idxs])

            real = stacked[:, 0:2].sum(axis=0)
            ratio = stacked[:, 2].sum(axis=0) / len(idxs)
            aug = stacked[:, 3:5].sum(axis=0)

            new_row = [class_name]
            new_row += list(real.astype("uint16"))
            new_row += ["{:.3f}".format(ratio)]
            new_row += list(aug.astype("uint16"))

            new_rows.append(new_row)

        summary_header = [
            ["Labels", "Train", "Valid", "Val/Train", "AugTrain", "AugValid"]
        ]
        summary = summary_header + new_rows

        print(tabulate(summary))

    def _summary(self, classes: List[str], label_path: Path):
        real = {}
        augmented = {}

        img_label_paths = YoloAugmentator.fileloader(label_path)
        for _, label_path in img_label_paths:
            labels = self._parse_labels(label_path)

            slabel_path = str(label_path)
            for label_value, *_ in labels:
                class_name = classes[label_value]
                if class_name not in augmented:
                    augmented[class_name] = 0
                    real[class_name] = 0

                augmented[class_name] += 1

                # is original
                if "_000_nflip_" in slabel_path and not "_checkered_" in slabel_path:
                    real[class_name] += 1

        return real, augmented

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
        return [self.calc_rotation(content) for content in yolo_content]

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


class ClassStripper:
    def __init__(
        self, label_dir, labels_to_remove, labels_and_files_to_remove, files_to_ignore
    ):
        self.label_dir = label_dir
        self.files_to_ignore = files_to_ignore

        with open(self.label_dir / "classes.txt", "r") as f:
            lines = f.readlines()
            self.old_classes = [cls_.strip() for cls_ in lines]

        self.labels_to_remove = labels_to_remove
        self.label_idxs_to_remove = [
            self.old_classes.index(label_to_remove)
            for label_to_remove in labels_to_remove
        ]

        self.labels_and_files_to_remove = labels_and_files_to_remove
        self.labels_and_files_idxs_to_remove = [
            self.old_classes.index(rem) for rem in labels_and_files_to_remove
        ]

        self.new_classes = list(self.old_classes)
        self.new_classes = [
            cls_
            for cls_ in self.new_classes
            if cls_ not in self.labels_to_remove
            and cls_ not in self.labels_and_files_to_remove
        ]

    def run(self):
        label_filenames = self.get_label_filenames()

        # remove all files if a label is in there
        for label_filename in label_filenames:
            with open(self.label_dir / label_filename, "r") as f:
                lines = f.readlines()

            # check if the label is present
            for line in lines:
                label_idx, _ = line.split(" ", 1)
                # remove all corresponding files
                if int(label_idx) in self.labels_and_files_idxs_to_remove:
                    img_filename = self.get_img_filename(label_filename)

                    os.remove(self.label_dir / label_filename)
                    os.remove(self.label_dir / img_filename)
                    break

        # requery after deleting
        label_filenames = self.get_label_filenames()

        # change the existing labels
        for label_filename in label_filenames:
            with open(self.label_dir / label_filename, "r") as f:
                lines = f.readlines()

            with open(self.label_dir / label_filename, "w") as f:
                for line in lines:
                    label_idx, bbox = line.split(" ", 1)
                    label_idx = int(label_idx)

                    # don't write the label back
                    if label_idx in self.label_idxs_to_remove:
                        continue

                    # get the labelname with the old_idx
                    label_name = self.old_classes[label_idx]

                    # obtain new index from new classes
                    new_idx = self.new_classes.index(label_name)

                    # write the new cls_idx back with the label
                    new_line = f"{new_idx} {bbox}"
                    f.write(new_line)

        with open(self.label_dir / "classes.txt", "r") as f:
            lines = f.readlines()

        with open(self.label_dir / "classes.txt", "w") as f:
            for line in lines:
                label_name = line.strip()

                # skip deleted classes
                if (
                    label_name in self.labels_to_remove
                    or label_name in self.labels_and_files_to_remove
                ):
                    continue

                f.write(line)

    def get_label_filenames(self):
        label_filenames = os.listdir(self.label_dir)
        label_filenames = [label for label in label_filenames if label.endswith(".txt")]
        label_filenames = [
            label for label in label_filenames if label not in self.files_to_ignore
        ]

        return label_filenames

    def get_img_filename(self, label_filename):
        name = os.path.splitext(label_filename)[0]
        files = os.listdir(self.label_dir)

        jpg = f"{name}.jpg"
        png = f"{name}.png"

        if jpg in files:
            return jpg

        elif png in files:
            return png
        else:
            raise ValueError(f"No img found for {label_filename}")


files_to_ignore = [
    "classes.names",
    "classes.txt",
    "train.txt",
    "val.txt",
    "val.txt.bak",
    "valid_yolo.txt",
    "train_yolo.txt",
    "log",
]


@click.command()
@click.argument(
    "target",
    # help="Values: <yolo/unet>"
    # "--target",
    # "-t",
    # multiple=False,
    # help="Values: <yolo/unet>"
)
def augment(target):
    if target == "yolo":
        augmentator = YoloAugmentator(
            config.train_dir,
            config.train_out_dir,
            config.valid_dir,
            config.valid_out_dir,
            config.merged_dir,
            config.augment.yolo.img_params,
            config.augment.label_transition_rotation,
            config.augment.label_transition_flip,
            clean=True,
        )
        augmentator.run()

        ClassStripper(
            config.train_out_dir,
            config.labels_to_remove,
            config.labels_and_files_to_remove,
            files_to_ignore,
        ).run()
        ClassStripper(
            config.valid_out_dir,
            config.labels_to_remove,
            config.labels_and_files_to_remove,
            files_to_ignore,
        ).run()

        augmentator.create_labels_file(augmentator.train_out_dir)
        augmentator.create_labels_file(augmentator.valid_out_dir)
        augmentator.summary()

    elif target == "unet":
        augmentator = UNetAugmentator(
            config.train_dir,
            config.train_out_dir,
            config.valid_dir,
            config.valid_out_dir,
            config.merged_dir,
            config.augment.unet.img_params,
            clean=True,
        )
        augmentator.run()


if __name__ == "__main__":
    augment()

    # if len(sys.argv) == 1 or sys.argv[1] == "train":
    #     print("Augmenting train files, this may take some time...")
    #     augmentator = YoloAugmentator(
    #         config.train_dir,
    #         config.train_out_dir
    #         files_to_ignore,
    #         label_transition_rotation,
    #         label_transition_flip,
    #         config.augment.perform_train,
    #         clean=True,
    #     )
    #     augmentator.run()

    #     if not config.augment.exclude_merged:
    #         print("Augmenting train_merged files, this may take some time...")
    #         augmentator = YoloAugmentator(
    #             config.merged_dir,
    #             config.train_out_dir
    #             files_to_ignore,
    #             label_transition_rotation,
    #             label_transition_flip,
    #             config.augment.perform_merged,
    #             clean=False,
    #         )
    #         augmentator.run()

    #     print("Stripping classes from train preprocessed...\n")
    #     ClassStripper(
    #         config.train_out_dir
    #         config.labels_to_remove,
    #         config.labels_and_files_to_remove,
    #         files_to_ignore,
    #     ).run()

    #     augmentator.create_train()
    #     augmentator.summary()

    # elif sys.argv[1] == "valid":
    #     print("Augmenting valid files, this may take some time...")
    #     augmentator = YoloAugmentator(
    #         config.valid_dir,
    #         config.valid_out_dir,
    #         files_to_ignore,
    #         label_transition_rotation,
    #         label_transition_flip,
    #         config.augment.perform_valid,
    #         clean=True,
    #     )
    #     augmentator.run()

    #     print("Stripping classes from valid_preprocessed...")
    #     ClassStripper(
    #         config.valid_out_dir,
    #         config.labels_to_remove,
    #         config.labels_and_files_to_remove,
    #         files_to_ignore,
    #     ).run()

    #     augmentator.create_train()
    #     augmentator.summary()

    # else:
    #     print("./augment.py <valid/train>")
