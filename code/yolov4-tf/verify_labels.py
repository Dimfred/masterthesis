#!/usr/bin/env python3

import cv2 as cv
import numpy as np
from yolov4.tf import YOLOv4

import sys
import os
from pathlib import Path

import utils
from utils import YoloBBox
from config import config


def write_class_name(img, text):
    img = cv.putText(
        img, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA
    )
    return img


if __name__ == "__main__":
    label_dir = Path(sys.argv[1])

    all_bboxes = []
    for file_ in os.listdir(label_dir):
        if not utils.is_img(file_):
            continue

        img_path = label_dir / file_
        label_path = utils.label_file_from_img(img_path)

        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        img = utils.resize_max_axis(img, 1200)

        labels = utils.load_ground_truth(label_path)

        bboxes = [YoloBBox(img.shape).from_ground_truth(gt) for gt in labels]

        # extract all bboxes from the img
        extracted_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.abs()
            extracted_bbox = cv.resize(img[y1:y2, x1:x2], (100, 100))
            extracted_bbox = np.pad(extracted_bbox, ((0, 10), (0, 10)))
            extracted_bboxes.append(extracted_bbox)

        full = [
            (bbox, ex_bbox, img_path) for bbox, ex_bbox in zip(bboxes, extracted_bboxes)
        ]
        all_bboxes.extend(full)

    all_bboxes = sorted(all_bboxes, key=lambda full: full[0].label)

    # just using the yolo class parser
    yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
    yolo.classes = config.yolo.classes

    # plot all bboxes
    cv.namedWindow("verify")
    base = np.zeros((1200, 1200, 3), dtype=np.uint8)

    yoff, xoff = 60, 10
    max_width = 10
    max_height = 10

    cur_cls_name = yolo.classes[all_bboxes[0][0].label]

    # split all classes into seperate containers
    class_wise = []
    _current_class_wise = []
    for bbox, ex_bbox, img_path in all_bboxes:
        cls_name = yolo.classes[bbox.label]

        if cls_name == cur_cls_name:
            # append this cls
            _current_class_wise.append((bbox, ex_bbox, img_path))
        else:
            # reset for next cls
            class_wise.append(_current_class_wise)
            _current_class_wise = [(bbox, ex_bbox, img_path)]
            cur_cls_name = cls_name

    samples_per_page = max_height * max_width

    # pad all classes with 0s if not full page
    for i, cls_ in enumerate(class_wise):
        should_cls = samples_per_page
        while should_cls < len(cls_):
            should_cls += samples_per_page

        pad_len = should_cls - len(cls_)
        class_wise[i].extend(
            [(None, np.zeros((110, 110), dtype=np.uint8), None) for _ in range(pad_len)]
        )

    cv.namedWindow("verify")
    # produce an image with all samples for each class
    for cls_ in class_wise:
        n_pages = int(len(cls_) / max_width / max_height)
        for i_page in range(n_pages):

            page = []
            for i_row in range(max_width):
                page_idx = lambda i: (i_page + i) * max_height * max_width

                start = i_row * max_width
                end = (i_row + 1) * max_width

                if i_page > 0:
                    start += samples_per_page
                    end += samples_per_page

                row = np.hstack([ex_bbox for _, ex_bbox, _ in cls_[start:end]])
                page.append(row)

            page = np.vstack(page).astype(np.uint8)
            ph, pw = page.shape[:2]

            yolo_bbox = cls_[0][0]
            cls_name = yolo.classes[yolo_bbox.label]

            show = base.copy()
            show = write_class_name(show, cls_name)
            show[yoff : yoff + ph, :pw] = page[..., np.newaxis]

            def mouse_cb(event, x, y, *args):
                if event == cv.EVENT_LBUTTONDOWN:
                    grid_x = x // 110
                    grid_y = (y - yoff) // 110

                    cls_idx = (
                        grid_x + (grid_y * max_height) + (i_page * samples_per_page)
                    )
                    yolo_bbox, ex_bbox, img_path = cls_[cls_idx]

                    print(utils.red(img_path))

            cv.setMouseCallback("verify", mouse_cb)
            cv.imshow("verify", np.uint8(show))

            def get_key():
                return 0xFF & cv.waitKey(10)

            key = get_key()
            while key != ord("q"):
                key = get_key()
