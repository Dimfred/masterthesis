#!/usr/bin/env python3.8

import cv2 as cv
import os


def get_labeled_files(label_path):
    files = os.listdir(label_path)

    labeled = []
    for f in files:
        if not f.endswith(".jpg"):
            continue

        name = f.split(".")[0]
        label_file = f"{name}.txt"

        if label_file not in files:
            continue

        labeled.append((f"{label_path}/{f}", f"{label_path}/{label_file}"))

    return labeled


def convert_coco(label_path, labeled_images):
    train_file = open(f"{label_path}/train_coco.txt", "w")

    for img_path, label_path in labeled_images:
        img = cv.imread(img_path)
        himg, wimg = img.shape[:2]

        with open(label_path, "r") as f:
            labels = f.readlines()

        labels = [line.strip() for line in labels]
        labels = [line.split(" ") for line in labels]

        coco_labels = [_convert_coco(himg, wimg, *yolo_label) for yolo_label in labels]
        coco_labels = " ".join(coco_labels)

        img_name = img_path.split("/")[-1]
        train_file.write(f"./data/preprocessed/{img_name} {coco_labels}\n")

        print(f"Converted: {img_name}")

    train_file.close()


def _convert_coco(himg, wimg, cls_, x, y, w, h):
    return f"{x},{y},{w},{h},{cls_}"
    # cls_, x, y, w, h = int(cls_), float(x), float(y), float(w), float(h)
    # will calculate absolute values
    # this is not needed for a converted_coco
    #xabs, yabs = wimg * x, himg * y
    #wabs, habs = wimg * w, himg * h

    #x1, y1 = int(xabs - 0.5 * wabs), int(yabs - 0.5 * habs)
    #x2, y2 = int(xabs + 0.5 * wabs), int(yabs + 0.5 * habs)

    #return f"{x1},{y1},{x2},{y2},{cls_}"


label_dir = "data/preprocessed"

if __name__ == "__main__":
    yolo_labels = get_labeled_files(label_dir)
    convert_coco(label_dir, yolo_labels)
