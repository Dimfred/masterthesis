#!/usr/bin/env python3.8
import albumentations as A

import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

import tfutils
from tfutils import GradientAccumulator, LossAccumulator, LearningRateScheduler
import utils

import numpy as np
import cv2 as cv

from tqdm import tqdm
import time
from tabulate import tabulate


def ffloat(f):
    return "{:.5f}".format(f)


# _callbacks = [
#     callbacks.LearningRateScheduler(lr_scheduler),
#     callbacks.TerminateOnNaN(),
#     callbacks.TensorBoard(log_dir="./log"),
#     SaveWeightsCallback(
#         yolo=yolo,
#         dir_path=config.yolo.checkpoint_dir,
#         weights_type=config.yolo.weights_type,
#         epoch_per_save=1,
#     ),
#     tfutils.BatchProgbarLogger(config.yolo.accumulation_steps),
# ]


class Trainer:
    def __init__(
        self,
        yolo,
        lr_scheduler=None,
        max_steps=1,
        validation_freq=2,
        accumulation_steps=1,
        map_after_steps=1,
        map_on_step_mod=1,
        lr=0.00026,
        burn_in=1000,
    ):
        self.yolo = yolo
        self.model = yolo.model

        self.step_counter = 0
        self.mini_step_counter = 0

        self.mAP = utils.MeanAveragePrecision(
            self.yolo.classes,
            self.yolo.input_size,
            iou_threshs=[0.5, 0.6, 0.7, 0.8],
        )

        self.max_steps = max_steps
        self.validation_freq = validation_freq
        self.lr_scheduler = LearningRateScheduler(self.model, lr_scheduler)
        self.accumulation_steps = accumulation_steps
        self.map_after_steps = map_after_steps
        self.map_on_step_mod = map_on_step_mod
        self.lr = lr
        self.burn_in = burn_in

        self.train_time = time.perf_counter()
        self.valid_time = time.perf_counter()
        self.train_time_start = time.perf_counter()

    def train(self, train_ds, valid_ds, **kwargs):
        trainable_vars = self.model.trainable_variables
        grad_accu = GradientAccumulator(self.accumulation_steps, trainable_vars)
        tloss_accu = LossAccumulator(self.accumulation_steps)

        base_lr = self.lr
        burn_in = self.burn_in


        for inputs, *labels, _ in train_ds:
            # training step
            step_grads, step_losses = self.train_step(inputs, labels)


            accumulated_losses = tloss_accu.accumulate(step_losses)
            accumulated_grads = grad_accu.accumulate(step_grads)
            if accumulated_grads is None:
                continue


            # apply optimization
            self.step_counter += 1
            if self.lr_scheduler:
                self.lr_scheduler(self.step_counter, base_lr, burn_in)

            self.model.optimizer.apply_gradients(accumulated_grads)

            ### TIME
            # start = time.perf_counter()
            self.print_train(accumulated_losses)
            # print("TOOK:", time.perf_counter() - start)

            # validation step
            if not self.is_validation_time():
                continue

            self.valid_time = time.perf_counter()

            n_valid_batches = np.ceil(len(valid_ds) / self.yolo.batch_size).astype(
                np.int32
            )
            vloss_accu = LossAccumulator(n_valid_batches)

            valid_ds_it = iter(valid_ds)
            vlosses = None

            self.mAP.reset()
            for _ in range(n_valid_batches):
                vinputs, *vlabels, idxs = next(valid_ds_it)
                voutputs, vlosses = self.valid_step(vinputs, vlabels)
                vlosses = vloss_accu.accumulate(vlosses)

                if self.is_map_time():
                    pred_batch = self.bboxes_from_outputs(voutputs)
                    label_batch = valid_ds.get_ground_truth(idxs)
                    self.mAP.add(pred_batch, label_batch, inverted_gt=True)

            if self.is_map_time():
                results = self.mAP.compute(show=False)
                tf.print(self.mAP.prettify(results))

            self.print_valid(vlosses)

            # TODO resize network?

    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            total_loss = 0
            losses = [None for _ in range(3)]

            output = self.model(inputs, training=True)
            for lidx, (o, l) in enumerate(zip(output, labels)):
                loss = self.model.loss(y_pred=o, y_true=l)
                total_loss += loss
                losses[lidx] = loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        return grads, tf.stack(losses)

    @tf.function
    def valid_step(self, inputs, labels):
        total_loss = 0
        losses = [0 for _ in range(3)]

        outputs = self.model(inputs, training=False)
        for lidx, (o, l) in enumerate(zip(outputs, labels)):
            loss = self.model.loss(y_pred=o, y_true=l)
            total_loss += loss
            losses[lidx] = loss

        return outputs, losses

    def is_validation_time(self):
        return (self.step_counter % self.validation_freq) == 0

    def is_map_time(self):
        min_map_steps_reached = self.step_counter >= self.map_after_steps
        is_valid_map_step = self.step_counter % self.map_on_step_mod == 0
        return min_map_steps_reached and is_valid_map_step
        # return True

    def bboxes_from_outputs(self, voutputs):
        # voutputs:
        # tuple(3)[(batch_size, *shape1), (batch_size, *shape2), (batch_size, *shape3)]
        candidates = self.yolo._transform_candidate_batch(voutputs)
        candidates = [candidate.numpy() for candidate in candidates]
        with tf.device("CPU:0"):
            bbox_batch = [
                self.yolo.candidates_to_pred_bboxes(
                    candidate[0],
                    iou_threshold=0.3,
                    score_threshold=0.25,
                )
                for candidate in candidates
            ]

        return bbox_batch

    def print_train(self, losses):
        took = ffloat(time.perf_counter() - self.train_time)
        loss_sum = ffloat(losses.sum())
        losses = (ffloat(l) for l in losses)

        # fmt: off
        p = [["Batch", "Took", "LossSum", "LossLarge", "LossMedium", "LossSmall", "Overall"]]
        p += [[self.step_counter, f"{took}s", loss_sum, *losses, self.overall_train_time]]
        print(tabulate(p))
        # fmt: on

        self.train_time = time.perf_counter()

    def print_valid(self, losses):
        took = ffloat(time.perf_counter() - self.valid_time)
        loss_sum = ffloat(losses.sum())
        losses = (ffloat(l) for l in losses)

        # fmt: off
        p = [[utils.green("Valid"), "Took", "LossSum", "LossLarge", "LossMedium", "LossSmall"]]
        p += [["", f"{took}s", loss_sum, *losses]]
        print(tabulate(p))
        # fmt: on

    @property
    def overall_train_time(self):
        import datetime

        took = time.perf_counter() - self.train_time_start
        return str(datetime.timedelta(seconds=took)).split(".")[0]
