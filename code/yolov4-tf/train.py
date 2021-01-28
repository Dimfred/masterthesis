# augmentation
import albumentations as A
import cv2 as cv
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import callbacks, optimizers
import time

AUTOTUNE = tf.data.experimental.AUTOTUNE

# model
from yolov4.tf import YOLOv4
from yolov4.tf.train import SaveWeightsCallback
from trainer import Trainer

# utils
import utils
from config import config

utils.seed("tf", "np", "imgaug")


def create_model():
    yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
    yolo.classes = config.yolo.classes
    yolo.input_size = config.yolo.input_size
    yolo.channels = config.yolo.channels
    yolo.batch_size = config.yolo.batch_size
    yolo.make_model()

    # yolo.load_weights(config.yolo.pretrained_weights, weights_type=config.yolo.weights_type)
    # yolo.load_weights(config.yolo.weights, weights_type=config.yolo.weights_type)

    return yolo


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
        # A.RandomScale(scale_limit=0.1, p=0.3),
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

def valid_augmentations(image, bboxes):
    _valid_augmentations = A.Compose([
        base_augmentations,
    ], bbox_params=A.BboxParams("yolo"))

    augmented = _valid_augmentations(image=image, bboxes=bboxes)
    return augmented["image"], augmented["bboxes"]
# fmt:on


if __name__ == "__main__":
    # model creation
    yolo = create_model()

    # optimizer = optimizers.Adam(learning_rate=config.yolo.lr)
    # optimizer = optimizers.SGD(
    #     learning_rate=config.yolo.lr, momentum=config.yolo.momentum
    # )
    optimizer = tfa.optimizers.SGDW(
        learning_rate=config.yolo.lr,
        momentum=config.yolo.momentum,
        weight_decay=config.yolo.decay,
    )

    yolo.compile(
        optimizer=optimizer,
        loss_iou_type=config.yolo.loss,
        loss_verbose=0,
        run_eagerly=config.yolo.run_eagerly,
        # steps_per_execution=config.yolo.accumulation_steps,
    )

    # dataset creation
    train_dataset = yolo.load_tfdataset(
        dataset_path=config.train_out_dir / "labels.txt",
        dataset_type=config.yolo.weights_type,
        label_smoothing=0.05,
        preload=config.yolo.preload_dataset,
        # preload=False,
        training=True,
        augmentations=train_augmentations,
    )

    # it = iter(train_dataset)
    # for i in range(10):
    #     item = next(it)
    #     img, labels = item
    #     print(img.shape, labels[0].shape)

    item = train_dataset._next_batch()
    # item = train_dataset._next()
    train_dataset.count = 0

    # build shapes
    x, l1, l2, l3 = item

    img_shapes = x.shape
    l1_shape = l1.shape
    l2_shape = l2.shape
    l3_shape = l3.shape
    label_shapes = (l1_shape, l2_shape, l3_shape)

    img_types = (np.float32, np.float32, np.float32, np.float32)
    label_type = (np.float32, np.float32, np.float32, np.float32, np.float32)
    label_types = tuple(label_type for _ in range(len(label_shapes)))

    # output_types=(img_types, *label_types)
    output_types = (np.float32, np.float32, np.float32, np.float32)
    output_shapes = (img_shapes, *label_shapes)

    for i, s in enumerate(output_shapes):
        print("shape", i, s)

    for i, t in enumerate(output_types):
        print("type", i, t)


    dataset = tf.data.Dataset.from_generator(
        train_dataset.generator,
        output_types=output_types,
        output_shapes=output_shapes,
    )
    # dataset = dataset.batch(config.yolo.batch_size)
    dataset = dataset.prefetch(20)

    # start_it = time.perf_counter()
    # for x, l1, l2, l3 in dataset:
    #     end_it = time.perf_counter()
    #     print("it took:", end_it - start_it)
    #     print("x", x.shape)
    #     print("l1", l1.shape)
    #     print("l2", l2.shape)
    #     print("l3", l3.shape)
    #     # THIS IS TRAINING OVERHEAD
    #     time.sleep(0.3)
    #     start_it = time.perf_counter()
    #     pass


    train_dataset = dataset

    valid_dataset = yolo.load_dataset(
        dataset_path=config.valid_out_dir / "labels.txt",
        dataset_type=config.yolo.weights_type,
        preload=config.yolo.preload_dataset,
        training=False,
        augmentations=valid_augmentations,
    )

    def lr_scheduler(step, lr):
        lr = config.yolo.lr
        max_steps = config.yolo.max_steps
        burn_in = config.yolo.burn_in

        if step < burn_in:
            return (lr / burn_in) * (step + 1)
        if step > 7000:
            return lr / 10
        if step > 5000:
            return lr / 5

        return lr

    # gib ihm
    trainer = Trainer(
        yolo,
        max_steps=config.yolo.max_steps,
        validation_freq=config.yolo.validation_freq,
        accumulation_steps=config.yolo.accumulation_steps,
        map_after_steps=config.yolo.map_after_steps,
        map_on_step_mod=config.yolo.map_on_step_mod,
        lr_scheduler=lr_scheduler,
    )
    trainer.train(train_dataset, valid_dataset)
