import cv2 as cv
import shutil as sh
from pathlib import Path

from config import config
import utils


def eval_file_from_img(img_path, type_):
    return config.eval_dir / f"{img_path.stem}_{type_}.csv"


def main(type_):
    eval_ds = utils.Yolo.load_dataset(config.eval_dir)
    for eval_img_path, eval_label_path in eval_ds:
        print("Creating for:", eval_img_path.name)

        eval_path = eval_file_from_img(eval_img_path, type_)
        # we have it already labeled
        if eval_path.exists():
            continue

        print(eval_path)

        inputs = []
        inp = 0
        while True:
            inp = input("Input matching idxs (text_arrow_idx, ecc_idx): ")
            inp = inp.strip()
            if inp == "q":
                break

            inputs.append(inp)

        with open(eval_path, "w") as f:
            for inp in inputs:
                print(inp, file=f)


if __name__ == "__main__":
    import sys
    if sys.argv[1] != "arrows" and sys.argv[1] != "texts":
        print(sys.argv[1], "not supported")
        sys.exit()

    main(sys.argv[1])
