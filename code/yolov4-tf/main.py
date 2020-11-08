import cv2 as cv
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


from yolov4.tf import YOLOv4

yolo = YOLOv4(tiny=True)

yolo.classes = "data/classes.names"
yolo.input_size = (416, 416)

yolo.make_model()
yolo.load_weights("weights/yolov4-tiny-custom_last.weights", weights_type="yolo")
yolo.inference(media_path="data/0_6.jpg") #, cv_frame_size=(600, 600))

# img = cv.imread("data/0_6.jpg")
# img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
# bboxs = yolo.predict(img)
# print(bboxs)
