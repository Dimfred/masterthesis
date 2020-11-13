#!/usr/bin/env python3
import sys
import os
import cv2 as cv

img_path = sys.argv[1]
orig = cv.imread(img_path)

counter = 0


def make_name(name, rotation, flip):
    global counter

    name, ext = os.path.splitext(name)
    # return "{}_{:03d}_{}{}".format(
    #     name, rotation, "hflip" if flip else "nflip", ext
    # )

    nname = f"{name}_{counter:02d}{ext}"
    counter += 1
    return nname


img = orig.copy()
for rot in (90, 180, 270):
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    cv.imwrite(make_name(img_path, rot, flip=False), img)


img = cv.flip(orig.copy(), +1)
cv.imwrite(make_name(img_path, 0, flip=True), img)

for rot in (90, 180, 270):
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    cv.imwrite(make_name(img_path, rot, flip=True), img)
