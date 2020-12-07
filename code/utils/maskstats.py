#!/usr/bin/env python3

import utils
from config import config

import os


if __name__ == "__main__":
    count = 0
    print("Files without a fg_mask")
    for file_ in sorted(os.listdir(config.train_dir)):
        if (
                utils.is_img(file_) 
                and not utils.has_mask(config.foregrounds_dir, file_)
                and not utils.has_unet_label(config.train_dir, file_)
                # TODO for valid dir also
        ):
            count += 1
            print(file_)

    print("Count:", count)
