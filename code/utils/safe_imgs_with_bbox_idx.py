import numpy as np
import cv2 as cv
import sys

from config import config
import utils


def main():
    ds_dir = config.eval_dir
    ds = utils.Yolo.load_dataset(ds_dir)

    for img_path, label_path in ds:
        img = cv.imread(str(img_path))
        labels = utils.Yolo.parse_labels(label_path)

        for idx, l in enumerate(labels):
            bbox = utils.YoloBBox(img.shape).from_ground_truth(l)

            x1, y1, x2, y2 = bbox.abs
            cv.putText(
                img, str(idx), (x1, y1 + 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        new_img_path = img_path.parent / f"{img_path.stem}_idxs{img_path.suffix}"
        cv.imwrite(str(new_img_path), img)


if __name__ == "__main__":
    main()
