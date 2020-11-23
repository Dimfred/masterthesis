#!/usr/bin/env python3

import os
import sys
from pathlib import Path


def has_image(filename, filenames):
    name, ext = os.path.splitext(filename)
    png = f"{name}.png"
    jpg = f"{name}.jpg"
    return png in filenames or jpg in filenames


def is_label_file(filename, filenames):
    return ".txt" in filename and has_image(filename, filenames) 


def get_label_files(dir_):
    filenames = os.listdir(dir_)
    filenames = [fn for fn in filenames if is_label_file(fn, filenames)]
    return filenames


def sort(fn):
    with open(fn, "r") as f:
        lines = f.readlines()

    lines = sorted(lines)

if __name__ == "__main__":
    dir_ = Path(sys.argv[1])
    label_files = get_label_files(dir_)
    for fn in label_files:
        sort(dir_ / fn)





