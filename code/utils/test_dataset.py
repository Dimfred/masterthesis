#!/usr/bin/env python3

import tensorflow as tf

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import cv2 as cv
import numpy as np
import os

def jpg_from_label(label_file):
    name = os.path.splitext(label_file)[0]
    return f"{name}.jpg"


def png_from_label(label_file):
    name = os.path.splitext(label_file)[0]
    return f"{name}.png"


def get_labels_and_img_names(label_dir):
    files = os.listdir(label_dir)
    txts_only = [f for f in files if ".txt" in f]

    labels_jpg = [
        (label, jpg_from_label(label))
        for label in txts_only
        if jpg_from_label(label) in files
    ]
    labels_png = [
        (label, png_from_label(label))
        for label in txts_only
        if png_from_label(label) in files
    ]
    return labels_jpg + labels_png


def generate_train(label_dir):
    labels_imgs = get_labels_and_img_names(label_dir)

    with open("test_dataset.txt", "w") as f:
        for _, img_name in labels_imgs:
            print(label_dir / img_name, file=f)

    return "test_dataset.txt"


def test_dataset(yolo, label_dir):
    from . import resize_max_axis

    train_txt = generate_train(label_dir)
    dataset = yolo.load_dataset(train_txt, training=False, dataset_type="yolo")

    for i, (images, gt) in enumerate(dataset):
        for j in range(len(images)):
            _candidates = []
            for candidate in gt:
                grid_size = candidate.shape[1:3]
                _candidates.append(
                    tf.reshape(
                        candidate[j], shape=(1, grid_size[0] * grid_size[1] * 3, -1)
                    )
                )
            candidates = np.concatenate(_candidates, axis=1)

            frame = images[j, ...] * 255
            frame = frame.astype(np.uint8)

            pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0])
            pred_bboxes = yolo.fit_pred_bboxes_to_original(pred_bboxes, frame.shape)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            image = resize_max_axis(frame, 1000)
            image = yolo.draw_bboxes(image, pred_bboxes)
            cv.namedWindow("result")
            cv.imshow("result", image)
            while cv.waitKey(10) & 0xFF != ord("q"):
                pass

        if i == 10:
            break

    cv.destroyWindow("result")
