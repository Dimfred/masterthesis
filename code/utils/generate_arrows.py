import cv2 as cv
import sys

import utils
from utils import Yolo, YoloBBox
from config import config

def main():
    if len(sys.argv) < 1:
        print("usage: gen / masks")
        sys.exit()

    if sys.argv[1] == "gen":
        arrow_img = config.train_dir / "23_00.jpg"
        arrow_labels = config.train_dir / "23_00.txt"

        img = cv.imread(str(arrow_img), cv.IMREAD_GRAYSCALE)
        labels = Yolo.parse_labels(arrow_labels)

        # utils.show(img)

        for i, label in enumerate(labels):
            bbox = YoloBBox(img.shape).from_ground_truth(label)
            x1, y1, x2, y2 = bbox.abs

            arrow_img = img[y1:y2, x1:x2]
            cv.imwrite(str(config.arrows_dir / f"{str(i).zfill(2)}.jpg"), arrow_img)
    elif sys.argv[1] == "masks":
        bg = 0
        probably_bg = 2

        all_ = config.arrows_dir.glob("**/*.jpg")
        mask_paths = [path for path in all_ if "mask" in str(path)]

        for mask_path in sorted(mask_paths):
            print(mask_path)
            img_path = utils.img_from_fg(config.arrows_dir, mask_path)

            img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

            print(img.shape)
            print(mask.shape)

            img[mask == bg] = 255
            img[mask == probably_bg] = 255
            # utils.show(img)

            new_img_path = img_path.parent / f"{img_path.stem}_arrow.jpg"
            cv.imwrite(str(new_img_path), img)



if __name__ == "__main__":
    main()
