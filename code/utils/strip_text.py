import cv2 as cv
import numpy as np
import os

from config import config
import utils
from utils import YoloBBox

from concurrent.futures import ThreadPoolExecutor
import shutil as sh


def is_annotated(path):
    return "_a_" in str(path)


counter = 0


def next_text_path():
    global counter
    path = config.texts_dir / f"{str(counter).zfill(3)}_text.png"
    counter += 1
    return str(path)


def main():
    for path in config.texts_dir.glob("**/*.*"):
        os.remove(path)

    classes = utils.Yolo.parse_classes(config.train_dir / "classes.txt")
    img_paths = utils.list_imgs(config.train_dir)
    fg_paths = list(config.foregrounds_dir.glob("**/*.*"))

    fgs_imgs_bboxes = []

    def read(fg_path):
        fg = cv.imread(str(fg_path))
        fg = fg[..., 0]
        fg[fg == 2] = 0
        fg[fg == 3] = 1

        img_path = utils.img_from_fg(config.train_dir, fg_path)
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        img = utils.resize_max_axis(img, size=1000)

        label_path = utils.Yolo.label_from_img(img_path)
        bboxes = [
            YoloBBox(img.shape).from_ground_truth(l)
            for l in utils.Yolo.parse_labels(label_path)
        ]

        fgs_imgs_bboxes.append((fg_path, fg, img_path, img, label_path, bboxes))

    with ThreadPoolExecutor(max_workers=32) as executor:
        for fg_path in fg_paths:
            if is_annotated(fg_path):
                executor.submit(read, fg_path)

    texts = []
    for fg_path, fg, img_path, img, label_path, bboxes in fgs_imgs_bboxes:
        mask = fg * img

        text_idxs = [
            idx for idx, bbox in enumerate(bboxes) if classes[bbox.label] == "text"
        ]

        for text_idx in text_idxs:
            is_okay = True

            text_bbox = bboxes[text_idx]
            for bbox_idx, bbox in enumerate(bboxes):
                if bbox_idx == text_idx:
                    continue

                if utils.calc_iou(bbox.abs, text_bbox.abs) > 0.0:
                    is_okay = False
                    break

            if is_okay:
                x1, y1, x2, y2 = text_bbox.abs
                text = mask[y1:y2, x1:x2]
                text[text == 0] = 255

                out_path = next_text_path()
                cv.imwrite(out_path, text)


if __name__ == "__main__":
    main()
