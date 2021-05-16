import pandas as pd
import numpy as np
import torch
import time

from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        data_loaders,
        loss,
        device,
        batch_size=32,
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

    def train(self, model, optimizer, n_steps):
        self.optimizer.zero_grad()

        self.overall_time = time.perf_counter()

        self.model = model
        self.model.train()

        self.optimizer = optimizer

        while True:
            for inputs, labels in self.train_ds:
                self.step_counter += 1

                if self.step_counter >= 1000:
                    self.early_stopping_counter += 1

                if self.lr_scheduler is not None:
                    lr = self.lr_scheduler(optimizer, self.step_counter)

                tloss = self.train_step(inputs, labels)
                self.train_summary_writer.add_scalar("Loss", tloss, self.step_counter)

                if self.step_counter % 10 == 0:
                    vinputs, vlabels = next(iter(self.valid_ds))
                    vloss, iou = self.valid_step(vinputs, vlabels)

                    self.valid_summary_writer.add_scalar(
                        "Loss", vloss, self.step_counter
                    )
                    self.valid_summary_writer.add_scalar("mIoU", iou, self.step_counter)

                    if iou > self.best_iou:
                        self.best_iou = iou
                        self.best_iou_step = self.step_counter
                        torch.save(
                            self.model.state_dict(), str(self.experiment.weights)
                        )

    def train_step(self, inputs, labels):
        subbatch_size = int(self.batch_size / self.subdivision)

        for subdiv in range(self.subdivision):
            start, end = subdiv * subbatch_size, (subdiv + 1) * subbatch_size
            sub_inputs, sub_labels = inputs[start:end], labels[start:end]

            sub_inputs = sub_inputs.to(self.device)
            sub_labels = sub_labels.to(self.device)

            pred = self.model(sub_inputs)
            loss = self.loss(pred, sub_labels)
            loss.backward()

        with torch.set_grad_enabled(True):
            self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item() / self.batch_size

    def valid_step(self, inputs, labels):
        self.model.eval()

        pred = self.model(inputs)
        loss = self.loss(pred, labels)

        with torch.set_grad_enabled(False):
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

        pred = self.combine_prediction(pred)
        iou = self.iou(labels, pred)

        return loss.item() / 23, iou  # self.batch_size

        # subbatch_size = int(self.batch_size / self.subdivision)

        # for subdiv in range(self.subdivision):
        #     start, end = subdiv * subbatch_size, (subdiv + 1) * subbatch_size
        #     sub_inputs, sub_labels = inputs[start:end], labels[start:end]

        #     sub_inputs = sub_inputs.to(self.device)
        #     sub_labels = sub_labels.to(self.device)

        #     pred = self.model(sub_inputs)
        #     loss = self.loss(pred, sub_labels)
        #     loss.backward()

        # with torch.set_grad_enabled(True):
        #     self.optimizer.step()
        # self.optimizer.zero_grad()

    def combine_prediction(prediction):
        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 1
        fg_pred = prediction[..., 0]
        bg_pred = prediction[..., 1]

        combined = (bg_pred + (1 - fg_pred)) / 2

        return combined

    def miou(self, target, prediction):
        intersection = torch.logical_and(target, prediction)
        union = torch.logical_or(target, prediction)
        iou_score = torch.sum(intersection, axis=(1, 2)) / torch.sum(union, axis=(1, 2))

        return iou_score.mean()
