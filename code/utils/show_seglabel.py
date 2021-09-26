from config import config
import utils

import cv2 as cv
import numpy as np

img = cv.imread(str(config.train_dir / "00_26_a.jpg"))
img = utils.resize_max_axis(img, 1000)
seglabel = np.load(config.train_dir / "00_26_a.npy")


seg_write = np.uint8(img * seglabel[...,np.newaxis])
cv.imwrite("munet_example.png", seg_write)


yolo_gt = utils.Yolo.label_from_img(config.train_dir / "00_26_a.jpg")
yolo_gt = utils.load_ground_truth(yolo_gt)
yolo_write = utils.show_bboxes(img, yolo_gt, type_="gt")
cv.imwrite("yolo_example.png", yolo_write)
