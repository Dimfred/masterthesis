import cv2 as cv
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from pathlib import Path
from yolov4.tf.train import SaveWeightsCallback

from utils.test_dataset import test_dataset

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from yolov4.tf import YOLOv4
from config import config

data = config.train_out_dir
checkpoints = Path("checkpoints")
weights = Path("weights")


#################
## YOLO PARAMS ##
#################

# TODO config
yolo = YOLOv4(tiny=True, small=True)
yolo.classes = str(data / "classes.txt")
yolo.input_size = (608, 608)
yolo.channels = 3
# TODO normally 64 and subdivision 3
yolo.batch_size = 16
# TODO check other params

yolo.make_model()
yolo.load_weights(weights / "yolov4-tiny-small.weights", weights_type="yolo")


#############
## DATASET ##
#############

train_dataset = yolo.load_dataset(
    dataset_path=str(data / "train_yolo.txt"),
    dataset_type="yolo",
    label_smoothing=0.05,
    # This enables augmentation. Mosaic, cut_out, mix_up are used
    training=True,
)
# test_dataset(yolo, dataset)

valid_dataset = yolo.load_dataset(
    dataset_path=str(data / "train_yolo.txt"), dataset_type="yolo", training=False
)

##############
## TRAINING ##
##############

# TODO config
epochs = 4000
lr = 1e-4


optimizer = optimizers.Adam(learning_rate=lr)
loss = "ciou"
yolo.compile(optimizer=optimizer, loss_iou_type=loss, loss_verbose=1)


def lr_scheduler(epoch):
    if epoch < int(epoch * 0.5):
        return lr
    if epoch < int(epochs * 0.8):
        return lr * 0.5
    if epoch < int(epochs * 0.9):
        return lr * 0.1

    return lr * 0.01


_callbacks = [
    callbacks.LearningRateScheduler(lr_scheduler),
    callbacks.TerminateOnNaN(),
    callbacks.TensorBoard(log_dir="./log"),
    SaveWeightsCallback(
        yolo=yolo, dir_path=checkpoints, weights_type="yolo", epoch_per_save=100
    ),
]

yolo.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    callbacks=_callbacks,
    validation_steps=1,
    validation_freq=1,
    steps_per_epoch=35,
)
