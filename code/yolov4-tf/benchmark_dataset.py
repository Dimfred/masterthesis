import tensorflow as tf
# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import time
import albumentations as A
import cv2 as cv

from yolov4.tf import YOLOv4
from yolov4.tf.train import SaveWeightsCallback
from config import config


def benchmark(name, dataset, epochs=3):
    start_time = time.perf_counter()
    for _ in range(epochs):
        inner_start = time.perf_counter()
        for i, _ in enumerate(dataset):
            tf.print(f"Took {name}", time.perf_counter() - inner_start())
            inner_start = time.perf_counter()

    tf.print(f"Took {name}", time.perf_counter - start_time)


yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
# yolo = YOLOv4()
# yolo = YOLOv4(tiny=config.yolo.tiny)
yolo.classes = config.yolo.classes
yolo.input_size = config.yolo.input_size
yolo.channels = config.yolo.channels
# TODO normally 64 and subdivision 3
yolo.batch_size = 16
# TODO check other params

yolo.make_model()

##############
## TRAINING ##
##############

# fmt:off
base_augmentations = A.Compose([
    A.PadIfNeeded(
        min_height=1000,
        min_width=1000,
        border_mode=cv.BORDER_CONSTANT,
        value=0,
        always_apply=True
    ),
    A.Resize(
        width=config.yolo.input_size,
        height=config.yolo.input_size,
        always_apply=True
    )
])

def train_augmentations(image, bboxes):
    _train_augmentations = A.Compose([
        # TODO tune params
        # TODO bboxed still disappearing
        # has to happen before the crop if not can happen that bboxes disappear
        A.Rotate(
            limit=10,
            border_mode=cv.BORDER_REFLECT_101,
            p=0.3,
        ),
        A.RandomScale(scale_limit=0.1, p=0.3),
        # THIS DOES NOT RESIZE ANYMORE THE RESIZING WAS COMMENTED OUT
        A.RandomSizedBBoxSafeCrop(
            width=None, # unused
            height=None, # unused
            p=0.3,
        ),
        A.OneOf([
            A.CLAHE(p=1),
            A.ColorJitter(p=1),
        ], p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.GaussNoise(p=0.3),
        base_augmentations
    ], bbox_params=A.BboxParams("yolo"))

    augmented = _train_augmentations(image=image, bboxes=bboxes)
    return augmented["image"], augmented["bboxes"]

# fmt:on


normal_ds = yolo.load_dataset(
    dataset_path=config.valid_out_dir / "labels.txt",
    dataset_type=config.yolo.weights_type,
    label_smoothing=0.05,
    preload=config.yolo.preload_dataset,
    training=True,
    augmentations=train_augmentations,
)

# for img, labels in normal_ds.generator():
#     for inner in labels:
#         print(inner.shape)

tf_ds = tf.data.Dataset.from_generator(
    normal_ds.generator,
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(
        (2, config.yolo.input_size, config.yolo.input_size, 1), # batch, y, x, chan
        (16, 76, 76, 3, 23), # labels
        (16, 38, 38, 3, 23), # labels
        (16, 19, 19, 3, 23), # labels
    )
)
# benchmark("NORMAL", tf_ds)

for sample in tf_ds.repeat().batch(2).take(10):
    print("yes")


# ds = yolo.load_tfdataset(
#     dataset_path=config.valid_out_dir / "labels.txt",
#     dataset_type=config.yolo.weights_type,
#     label_smoothing=0.05,
#     preload=config.yolo.preload_dataset,
#     training=True,
#     augmentations=train_augmentations,
# )
