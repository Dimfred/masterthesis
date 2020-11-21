#!/usr/bin/env python3

from grabcut3 import GrabCut

from config import config

dir_ = config.label_dir


if __name__ == "__main__":
    gc = GrabCut()
    gc.load_image("05_00.jpg")
    gc.matte()
