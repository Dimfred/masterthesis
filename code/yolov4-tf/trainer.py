#!/usr/bin/env python3.8

import tensorflow as tf
import numpy as np
import cv2 as cv
import tensorflow_datasets as tfds
from functools import partial
import albumentations as A

AUTOTUNE = tf.data.experimental.AUTOTUNE

print(tfds.__version__)
