import pandas as pd
import numpy as np
import torch
import time
from tabulate import tabulate
import numba as nb

import utils


from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


def ffloat(f):
    return "{:.5f}".format(f)


def bstart():
    return time.perf_counter()


def bend(tstart, name="none"):
    t = time.perf_counter() - tstart
    print(f"Took {name}", ffloat(t))


class Trainer:
    def __init__(
        self,
        data_loaders,
        loss,
        device,
        max_steps=2000,
        batch_size=32,
        valid_batch_size=32,
        subdivision=1,
        valid_subdivision=1,
        lr_scheduler=None,
        experiment=None,
        loss_type=None,
    ):
        self.device = device

        # config
        self.train_ds, self.valid_ds = data_loaders
        self.loss = loss
        self.loss_type = loss_type

        self.batch_size = batch_size
        assert self.batch_size % 2 == 0
        self.subdivision = subdivision
        assert subdivision > 0
        self.valid_batch_size = valid_batch_size
        self.valid_subdivision = valid_subdivision

        self.lr_scheduler = lr_scheduler

        # runtime
        self.step_counter = 0
        self.early_stopping = 1500
        self.early_stopping_counter = 0
        self.max_steps = max_steps

        # logging
        self.experiment = experiment
        self.train_summary_writer = SummaryWriter(self.experiment.tb_train_dir)
        self.valid_summary_writer = SummaryWriter(self.experiment.tb_valid_dir)

        # best model
        # self.best_iou = 0
        self.best_f1 = 0
        self.best_step = 0

        # time
        self.train_time = 0
        self.valid_time = 0
        self.overall_time = 0


    def train(self, model, optimizer):
        self.overall_time = time.perf_counter()

        self.optimizer = optimizer

        self.model = model

        done = False
        while not done:
            for inputs, labels in self.train_ds:
                self.step_counter += 1
                self.early_stopping_counter += 1

                if self.step_counter > self.max_steps:
                    done = True
                    break

                if self.early_stopping_counter >= self.early_stopping:
                    done = True
                    break

                if self.lr_scheduler is not None:
                    lr = self.lr_scheduler(optimizer, self.step_counter)
                    # print(lr)

                tloss = self.train_step(inputs, labels)
                self.print_train(tloss)

                # if tloss < 1e-5:
                #     self.on_nan_or_zero(inputs, labels)

                if self.step_counter % 1 == 0:
                    vinputs, vlabels = next(iter(self.valid_ds))

                    vloss, preds = self.valid_step(vinputs, vlabels)
                    vlabels = vlabels.cpu().numpy()

                    if self.loss_type == "focal" or self.loss_type == "dice":
                        preds = [nn.Softmax(dim=1)(pred) for pred in preds]
                    else: # binfocal
                        preds = [nn.Sigmoid()(pred) for pred in preds]

                    preds = [pred.cpu().numpy() for pred in preds]
                    preds = np.concatenate(preds, axis=0)

                    preds = self.combine_prediction(preds)
                    iou, recall, precision, f1 = self.metrics(vlabels, preds)

                    self.print_valid(vloss, iou, recall, precision, f1)

                    vinputs = vinputs.cpu().numpy()

                    # if iou > self.best_iou:
                    if f1 > self.best_f1:
                        self.early_stopping_counter = 0

                        # self.best_iou = iou
                        self.best_f1 = f1
                        self.best_step = self.step_counter
                        torch.save(
                            self.model.state_dict(), str(self.experiment.weights)
                        )

                        # if self.step_counter > 600:
                        #     for img, pred in zip(vinputs, pred):
                        #         img = np.uint8(img * 255)
                        #         pred = np.uint8(pred * 255)
                        #         utils.show(
                        #             img.transpose((1, 2, 0)),
                        #             np.expand_dims(pred, axis=2),
                        #         )


        with open(self.experiment.results, "w") as f:
            print(self.best_f1, file=f)

    def on_nan_or_zero(self, inputs, labels):
        self.model.eval()

        # inputs, labels = inputs.cpu().numpy(), labels.cpu().numpy()
        for img, mask in zip(inputs, labels):
            with torch.no_grad():
                pred = self.model(torch.unsqueeze(img.to(self.device), axis=0))[0]

            img, labels, pred = (
                img.cpu().numpy(),
                mask.cpu().numpy(),
                pred.cpu().numpy(),
            )

            img = np.uint8(img * 255).transpose((1, 2, 0))
            mask = np.expand_dims(np.uint8(mask * 255), axis=2)

            pred = pred[1]
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = np.expand_dims(np.uint8(pred * 255), axis=2)

            utils.show(img, mask, pred)

    def train_step(self, inputs, labels):
        self.model.train()
        self.optimizer.zero_grad()

        subbatch_size = int(self.batch_size / self.subdivision)

        n_subbatches = np.ceil(len(inputs) / self.subdivision)

        total_loss = 0
        with torch.set_grad_enabled(True):
            for subdiv in range(self.subdivision):
                start, end = subdiv * subbatch_size, (subdiv + 1) * subbatch_size
                sub_inputs, sub_labels = inputs[start:end], labels[start:end]

                if sub_inputs.shape[0] == 0:
                    break

                sub_inputs = sub_inputs.to(self.device)
                sub_labels = sub_labels.to(self.device)

                pred = self.model(sub_inputs)
                loss = self.loss(pred, sub_labels) / n_subbatches
                loss.backward()
                total_loss += loss.item()

            self.optimizer.step()
        self.optimizer.zero_grad()

        # return loss.item() / self.batch_size
        # return total_loss / self.subdivision
        return total_loss

    def print_train(self, loss):
        took = ffloat(time.perf_counter() - self.train_time)

        self.train_summary_writer.add_scalar("Loss", loss, self.step_counter)

        pretty = [["Step", "Took", "Loss", "Overall"]]
        pretty += [
            [
                self.step_counter,
                f"{took}s",
                ffloat(loss * 1000),
                self.overall_train_time,
            ]
        ]
        pretty += [["Experiment", self.experiment.run]]
        # pretty += [["BestIoU", f"{ffloat(self.best_iou*100)}%"]]
        pretty += [["BestF1", f"{ffloat(self.best_f1*100)}%"]]
        pretty += [["BestStep", self.best_step]]
        print(tabulate(pretty))

        self.train_time = time.perf_counter()

    @property
    def overall_train_time(self):
        import datetime

        took = time.perf_counter() - self.overall_time
        return str(datetime.timedelta(seconds=took)).split(".")[0]

    def valid_step(self, inputs, labels):
        self.model.eval()
        self.optimizer.zero_grad()

        subbatch_size = int(self.valid_batch_size / self.valid_subdivision)
        total_loss = 0

        preds = []
        with torch.no_grad():
            for subdiv in range(self.valid_subdivision):
                start, end = subdiv * subbatch_size, (subdiv + 1) * subbatch_size
                sub_inputs, sub_labels = inputs[start:end], labels[start:end]

                sub_inputs = sub_inputs.to(self.device)
                sub_labels = sub_labels.to(self.device)

                pred = self.model(sub_inputs)

                loss = self.loss(pred, sub_labels) / self.valid_subdivision
                total_loss += loss.item()

                preds.append(pred)

        return total_loss, preds

    def print_valid(self, loss, iou, recall, precision, f1):
        self.valid_summary_writer.add_scalar("Loss", loss, self.step_counter)
        self.valid_summary_writer.add_scalar("mIoU", iou, self.step_counter)
        self.valid_summary_writer.add_scalar("Recall", recall, self.step_counter)
        self.valid_summary_writer.add_scalar("Precision", precision, self.step_counter)
        self.valid_summary_writer.add_scalar("F1", f1, self.step_counter)

        pretty = [["Loss", "mIoU", "Recall", "Precision", "F1"]]
        pretty += [
            [
                utils.green(ffloat(loss * 1000)),
                f"{utils.green(ffloat(iou * 100))}%",
                f"{utils.green(ffloat(recall * 100))}%",
                f"{utils.green(ffloat(precision * 100))}%",
                f"{utils.green(ffloat(f1 * 100))}%",
            ]
        ]
        print(tabulate(pretty))

    def combine_prediction(self, prediction):
        if self.loss_type == "focal":
            bg_pred = prediction[:, 0]
            fg_pred = prediction[:, 1]

            prediction = (fg_pred + (1 - bg_pred)) / 2
            # combined = fg_pred

        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 1

        return prediction

    def metrics(self, target, prediction):
        iou = self.miou(target, prediction)

        tps, fps, fns = tps_fps_fns(target, prediction)
        recall = self.recall(tps, fns)
        precision = self.precision(tps, fps)
        f1 = self.f1(precision, recall)

        return iou, recall, precision, f1

    def miou(self, target, prediction):
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection, axis=(1, 2)) / np.sum(union, axis=(1, 2))

        return iou_score.mean()

    def recall(self, tp, fn):
        denom = (tp + fn)
        if denom == 0:
            denom = 1
        return tp / denom

    def precision(self, tp, fp):
        denom = (tp + fp)
        if denom == 0:
            denom = 1
        return tp / denom

    def f1(self, precision, recall):
        denom = (precision + recall)
        if denom == 0:
            denom = 1
        return 2 * precision * recall / denom

@nb.njit
def tps_fps_fns(target, prediction):
    tps, fps, fns = 0, 0, 0
    for target_, prediction_ in zip(target, prediction):
        for target_row, prediction_row in zip(target_, prediction_):
            for target_val, prediction_val in zip(target_row, prediction_row):
                if target_val == 1:
                    if prediction_val == 1:
                        tps += 1
                    else:
                        fns += 1
                else:
                    if prediction_val == 1:
                        fps += 1

    return tps, fps, fns
