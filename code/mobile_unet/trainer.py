import pandas as pd
import numpy as np
import torch
import time
from tabulate import tabulate

import utils

from torch.utils.tensorboard import SummaryWriter


def ffloat(f):
    return "{:.5f}".format(f)


class Trainer:
    def __init__(
        self,
        data_loaders,
        loss,
        device,
        batch_size=32,
        valid_batch_size=32,
        subdivision=1,
        lr_scheduler=None,
        experiment=None,
    ):
        self.device = device

        # config
        self.train_ds, self.valid_ds = data_loaders
        self.loss = loss

        self.batch_size = batch_size
        assert self.batch_size % 2 == 0
        self.subdivision = subdivision
        assert subdivision > 0
        self.valid_batch_size = valid_batch_size

        self.lr_scheduler = lr_scheduler

        # runtime
        self.step_counter = 0
        self.early_stopping = 1000
        self.early_stopping_counter = 0

        # logging
        self.experiment = experiment
        self.train_summary_writer = SummaryWriter(self.experiment.tb_train_dir)
        self.valid_summary_writer = SummaryWriter(self.experiment.tb_valid_dir)

        # best model
        self.best_iou = 0
        self.best_iou_step = 0

        # time
        self.train_time = 0
        self.valid_time = 0
        self.overall_time = 0

    def train(self, model, optimizer):
        self.overall_time = time.perf_counter()

        self.optimizer = optimizer

        self.model = model
        self.model.train()

        while True:
            for inputs, labels in self.train_ds:
                self.step_counter += 1

                if self.step_counter >= 1000:
                    self.early_stopping_counter += 1

                if self.lr_scheduler is not None:
                    lr = self.lr_scheduler(optimizer, self.step_counter)

                tloss = self.train_step(inputs, labels)
                self.print_train(tloss)

                if self.step_counter % 10 == 0:
                    vinputs, vlabels = next(iter(self.valid_ds))
                    vloss, pred = self.valid_step(vinputs, vlabels)

                    pred, vlabels = pred.cpu().numpy(), vlabels.cpu().numpy()
                    pred = self.combine_prediction(pred)
                    iou = self.miou(vlabels, pred)

                    self.print_valid(vloss, iou)

                    if iou > self.best_iou:
                        self.best_iou = iou
                        self.best_iou_step = self.step_counter
                        torch.save(
                            self.model.state_dict(), str(self.experiment.weights)
                        )

    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()

        subbatch_size = int(self.batch_size / self.subdivision)

        with torch.set_grad_enabled(True):
            for subdiv in range(self.subdivision):
                start, end = subdiv * subbatch_size, (subdiv + 1) * subbatch_size
                sub_inputs, sub_labels = inputs[start:end], labels[start:end]

                # sub_inputs = sub_inputs.cuda()
                # sub_labels = sub_labels.cuda()

                sub_inputs = sub_inputs.to(self.device)
                sub_labels = sub_labels.to(self.device)

                pred = self.model(sub_inputs)
                loss = self.loss(pred, sub_labels)
                loss.backward()

            self.optimizer.step()

        return loss.item() / self.batch_size

    def print_train(self, loss):
        took = ffloat(time.perf_counter() - self.train_time)

        self.train_summary_writer.add_scalar("Loss", loss, self.step_counter)

        pretty = [["Step", "Took", "Loss", "Overall"]]
        pretty += [[self.step_counter, f"{took}s", loss, self.overall_train_time]]
        pretty += [["Experiment", self.experiment.run]]
        pretty += [["BesIoU", ffloat(self.best_iou*100)]]
        pretty += [["BestIoUStep", self.best_iou_step]]
        print(tabulate(pretty))

        self.train_time = time.perf_counter()

    @property
    def overall_train_time(self):
        import datetime

        took = time.perf_counter() - self.overall_time
        return str(datetime.timedelta(seconds=took)).split(".")[0]

    def valid_step(self, inputs, labels):
        self.optimizer.zero_grad()
        with torch.no_grad():
            inputs = inputs.cuda()
            labels = labels.cuda()

            pred = self.model(inputs)
            loss = self.loss(pred, labels)

        return loss.item() / self.valid_batch_size, pred

    def print_valid(self, loss, iou):
        self.valid_summary_writer.add_scalar("Loss", loss, self.step_counter)
        self.valid_summary_writer.add_scalar("mIoU", iou, self.step_counter)

        pretty = [["Loss", "mIoU"]]
        pretty += [[utils.green(loss), utils.green(ffloat(iou * 100))]]
        print(tabulate(pretty))

    def combine_prediction(self, prediction):
        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 1
        bg_pred = prediction[:, 0]
        fg_pred = prediction[:, 1]

        combined = (fg_pred + (1 - bg_pred)) / 2

        return combined

    def miou(self, target, prediction):
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection, axis=(1, 2)) / np.sum(union, axis=(1, 2))

        return iou_score.mean()
