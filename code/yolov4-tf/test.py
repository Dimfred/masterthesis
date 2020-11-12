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
size = 736
yolo.input_size = (size, size)
yolo.channels = 1

# default 0.25, 0.3
inference_params = {"score_threshold": 0.8, "iou_threshold": 0.8}

yolo.make_model()

# tiny
# yolo.load_weights("weights/tiny_custom_last.weights", weights_type="yolo")

# small
yolo.load_weights("weights/v1sm/small_conf_best.weights", weights_type="yolo")


# yolo.load_weights(
#     "weights/tiny-channel3-burnin100-in416x416-noangle-sat1.5-expos1.5-0.1hue/yolov4-tiny-custom_last.weights",
#     weights_type="yolo",
# )

#yolo.inference(media_path="data/unlabeled/05_00.jpg")
#yolo.inference(media_path="data/unlabeled/03_00.png")
#yolo.inference(media_path="data/unlabeled/03_01.png")
#yolo.inference(media_path="data/unlabeled/03_02.png")
#yolo.inference(media_path="data/unlabeled/03_03.png")
#yolo.inference(media_path="data/unlabeled/03_04.png")
#yolo.inference(media_path="data/unlabeled/03_05.png")
#yolo.inference(media_path="data/unlabeled/03_06.png")
#yolo.inference(media_path="data/unlabeled/03_07.png")
#yolo.inference(media_path="data/unlabeled/03_08.png")
#yolo.inference(media_path="data/unlabeled/03_09.png")

yolo.inference(media_path="data/unlabeled/01_00.jpg")
#yolo.inference(media_path="data/labeled/00_01.jpg")
#yolo.inference(media_path="data/labeled/00_02.jpg")
#yolo.inference(media_path="data/labeled/00_03.jpg")
#yolo.inference(media_path="data/labeled/00_04.jpg")
#yolo.inference(media_path="data/unlabeled/00_07.jpg")
# yolo.inference(media_path="data/unlabeled/01_00.jpg")
# yolo.inference(media_path="data/unlabeled/02_00.jpg")
# yolo.inference(media_path="data/unlabeled/02_01.jpg")
# yolo.inference(media_path="data/unlabeled/02_02.jpg")
# yolo.inference(media_path="data/unlabeled/02_03.jpg")
# yolo.inference(media_path="data/unlabeled/oesi_0.jpg")

# img = cv.imread("data/0_6.jpg")
# img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
# bboxs = yolo.predict(img)
# print(bboxs)
