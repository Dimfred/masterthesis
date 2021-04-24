import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import utils
from config import config
from augment import YoloAugmentator

from tabulate import tabulate
import click


def count_if(container, condition):
    return sum(1 for ele in container if condition(ele))


class SummaryWriter:
    def __init__(self, train_out_dir, valid_out_dir, test_out_dir):
        self.train_out_dir = train_out_dir
        self.valid_out_dir = valid_out_dir
        self.test_out_dir = test_out_dir

    def summary(self):
        self.cls_summary()
        self.file_summary()

    def file_summary(self):
        train_sum = self._file_summary(self.train_out_dir)
        valid_sum = self._file_summary(self.valid_out_dir)
        test_sum = self._file_summary(self.test_out_dir)

        print("--------------------------------------------------------")
        print("TRAIN")
        print(train_sum)
        print("--------------------------------------------------------")

        print("--------------------------------------------------------")
        print("VALID")
        print(valid_sum)
        print("--------------------------------------------------------")

        print("--------------------------------------------------------")
        print("TEST")
        print(test_sum)
        print("--------------------------------------------------------")

    def _file_summary(self, dir_):
        imgs = utils.list_imgs(dir_)

        # first without augmentation only
        pure = count_if(
            imgs,
            lambda path: (
                self._is_original(path)
                and not self._is_checkered(path)
                and not self._is_annotated(path)
            )
        )

        checkered = count_if(
            imgs,
            lambda path: (
                self._is_original(path)
                and self._is_checkered(path)
                and not self._is_annotated(path)
            )

        )

        annotated = count_if(
            imgs,
            lambda path: (
                self._is_original(path)
                and not self._is_checkered(path)
                and self._is_annotated(path)
            )
        )

        checkered_and_annotated = count_if(
            imgs,
            lambda path: (
                self._is_original(path)
                and self._is_checkered(path)
                and self._is_annotated(path)
            )
        )

        n_persons = len(set(path.name[:2] for path in imgs))

        originals = [pure, checkered, annotated, checkered_and_annotated]

        # all files
        pure = count_if(
            imgs,
            lambda path: (
                not self._is_checkered(path)
                and not self._is_annotated(path)
            )
        )

        checkered = count_if(
            imgs,
            lambda path: (
                self._is_checkered(path)
                and not self._is_annotated(path)
            )

        )

        annotated = count_if(
            imgs,
            lambda path: (
                not self._is_checkered(path)
                and self._is_annotated(path)
            )
        )

        checkered_and_annotated = count_if(
            imgs,
            lambda path: (
                self._is_checkered(path)
                and self._is_annotated(path)
            )
        )

        augmented = [pure, checkered, annotated, checkered_and_annotated]

        pretty = [["", "N", "C", "A", "CA", "NumImgs", "NumPersons"]]
        pretty += [["Originals", *originals, sum(originals), n_persons]]
        pretty += [["Augmented", *augmented, sum(augmented)]]

        return tabulate(pretty)

    def cls_summary(self):
        # reparse classes after stripper hits
        class_names = utils.Yolo.parse_classes(self.train_out_dir / "classes.txt")

        real_train, aug_train = self._summary(class_names, self.train_out_dir)
        real_valid, aug_valid = self._summary(class_names, self.valid_out_dir)
        real_test, aug_test = self._summary(class_names, self.test_out_dir)

        # class_names = sorted(real_train.keys())
        summary_count = np.vstack(
            [
                (
                    real_train[class_name],
                    real_valid.get(class_name, 0),
                    real_test.get(class_name, 0),
                    aug_train[class_name],
                    aug_valid.get(class_name, 0),
                    aug_test.get(class_name, 0),
                )
                for class_name in class_names
            ]
        )

        trains = summary_count[:, 0]
        valids = summary_count[:, 1]
        tests = summary_count[:, 2]

        # TODO kill that
        valid_to_train_ratio = valids / trains
        test_to_train_ratio = tests / trains

        reals = summary_count[:, :3].copy()
        augs = summary_count[:, 3:].copy()
        summary_count = np.append(reals, valid_to_train_ratio[:, np.newaxis], axis=1)
        summary_count = np.append(
            summary_count, test_to_train_ratio[:, np.newaxis], axis=1
        )
        summary_count = np.append(summary_count, augs, axis=1)

        # accumulate summary idxs with common base class
        # idxs_to_combine = {}
        # for idx, sub_class_name in enumerate(class_names):
        #     class_name, *_ = sub_class_name.split("_", 1)

        #     if class_name not in idxs_to_combine:
        #         idxs_to_combine[class_name] = []

        #     idxs_to_combine[class_name].append(idx)

        # merge base classes
        # new_class_names = sorted(idxs_to_combine.keys())
        # new_rows = []
        # for class_name in new_class_names:
        #     idxs = idxs_to_combine[class_name]
        #     stacked = np.vstack([summary_count[idx] for idx in idxs])

        #     real = stacked[:, 0:3].sum(axis=0)
        #     train, valid, test = real[0:3]
        #     valid_train_ratio = valid / train * 100
        #     test_train_ratio = test / train * 100
        #     aug = stacked[:, 5:8].sum(axis=0)

        #     new_row = [class_name]
        #     new_row += list(real.astype("uint16"))
        #     new_row += ["{:.3f}%".format(valid_train_ratio)]
        #     new_row += ["{:.3f}%".format(test_train_ratio)]
        #     new_row += list(aug.astype("uint16"))

        #     new_rows.append(new_row)

        pretty = []
        for cls_name, row in zip(class_names, summary_count):
            pretty_row = [cls_name]
            pretty_row += [int(n) for n in row[0:3]]
            pretty_row += ["{:.2f}%".format(f * 100) for f in row[3:5]]
            pretty_row += [int(n) for n in row[5:8]]
            pretty += [pretty_row]


        summary_header = [
            [
                "Labels",
                "Train",
                "Valid",
                "Test",
                "Val/Train",
                "Test/Train",
                "AugTrain",
                "AugValid",
                "AugTest",
            ]
        ]
        summary = summary_header + list(pretty)

        print(tabulate(summary))



    def _is_original(self, path):
        return "000_nflip" in path.name and not "checkered" in path.name

    def _is_checkered(self, path):
        return "_c_" in path.name or "checkered" in path.name

    def _is_annotated(self, path):
        return "_a_" in path.name

    def _summary(self, classes: List[str], label_path: Path):
        img_label_paths = YoloAugmentator.fileloader(label_path)

        spath_and_labels = []

        def load_and_store(path):
            labels = utils.Yolo.parse_labels(path)
            spath_and_labels.append((str(path), labels))

        with ThreadPoolExecutor(max_workers=32) as executor:
            for _, label_path in img_label_paths:
                executor.submit(load_and_store, label_path)

        real = {}
        augmented = {}
        for spath, labels in spath_and_labels:
            for label_value, *_ in labels:
                class_name = classes[int(label_value)]
                if class_name not in augmented:
                    augmented[class_name] = 0
                    real[class_name] = 0

                augmented[class_name] += 1

                # is original
                if "_000_nflip_" in spath and not "_checkered_" in spath:
                    real[class_name] += 1

        return real, augmented


@click.command()
@click.argument("target")
def main(target):
    if target == "yolo":
        writer = SummaryWriter(
            config.train_out_dir, config.valid_out_dir, config.test_out_dir
        )
        writer.summary()
    elif target == "unet":
        pass


if __name__ == "__main__":
    main()
