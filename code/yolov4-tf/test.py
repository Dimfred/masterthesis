#!/usr/bin/env python3

import cv2 as cv
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


from yolov4.tf import YOLOv4

# small will use yolov4 head with 3 yolo layers
yolo = YOLOv4(tiny=True, small=True)

# real classes
#yolo.classes = "data/preprocessed/classes.txt"
# used classes
yolo.classes = "trained_classes.txt"
# 608, 736, 832, 928, 960, 1120, 1280, 1600
size = 608
yolo.input_size = (size, size)
yolo.channels = 1

# default 0.25, 0.3
inference_params = {"score_threshold": 0.8, "iou_threshold": 0.8}

yolo.make_model()

# tiny
# yolo.load_weights("weights/tiny_custom_last.weights", weights_type="yolo")

# small
yolo.load_weights("weights/label.weights", weights_type="yolo")

# juli
yolo.inference(media_path="data/labeled/06_00.jpg")
yolo.inference(media_path="data/labeled/06_01.jpg")
yolo.inference(media_path="data/labeled/06_02.jpg")

# grounds, sources, currents, inductors
#yolo.inference(media_path="data/labeled/00_08.jpg")
#yolo.inference(media_path="data/labeled/00_09.jpg")
#yolo.inference(media_path="data/labeled/00_10.jpg")

# valid
yolo.inference(media_path="data/valid/00_11.jpg")
yolo.inference(media_path="data/valid/00_11_00.jpg")
yolo.inference(media_path="data/valid/00_11_01.jpg")
yolo.inference(media_path="data/valid/00_11_02.jpg")
yolo.inference(media_path="data/valid/00_11_03.jpg")
yolo.inference(media_path="data/valid/00_11_04.jpg")
yolo.inference(media_path="data/valid/00_11_05.jpg")
yolo.inference(media_path="data/valid/00_11_06.jpg")


#yolo.inference(media_path="data/unlabeled/03_02.png")
#yolo.inference(media_path="data/unlabeled/03_03.png")
#yolo.inference(media_path="data/unlabeled/03_04.png")
#yolo.inference(media_path="data/unlabeled/03_05.png")
#yolo.inference(media_path="data/unlabeled/03_06.png")
#yolo.inference(media_path="data/unlabeled/03_07.png")
#yolo.inference(media_path="data/unlabeled/03_08.png")
#yolo.inference(media_path="data/unlabeled/03_09.png")

#yolo.inference(media_path="data/labeled/03_04.png")
#yolo.inference(media_path="data/labeled/03_05.png")
#yolo.inference(media_path="data/labeled/03_06.png")
#yolo.inference(media_path="data/labeled/03_07.png")
#yolo.inference(media_path="data/labeled/03_08.png")
#yolo.inference(media_path="data/labeled/03_09.png")


# img = cv.imread("data/0_6.jpg")
# img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
# bboxs = yolo.predict(img)
# print(bboxs)
