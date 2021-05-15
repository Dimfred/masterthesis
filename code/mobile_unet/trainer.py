import pandas as pd
import numpy as np
import torch

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
        experiment=None
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


    def train(self, model, optimizer, n_steps):
        self.model = model
        self.optimizer = optimizer

        while True:
            for inputs, labels in self.train_ds:
                self.step_counter += 1

                if self.step_counter >= 1000:
                    self.early_stopping_counter += 1

                if self.lr_scheduler is not None:
                    lr = self.lr_scheduler(optimizer, self.step_counter)

                loss = self.train_step(inputs, labels)







        for step in range(1, n_steps + 1):
            train_epoch_loss = self._train_on_epoch(model, optimizer, epoch)
            val_epoch_loss = self._val_on_epoch(model, optimizer)

            hist = {
                "epoch": epoch,
                "train_loss": train_epoch_loss,
                "val_loss": val_epoch_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            self.history.append(hist)

            if self.on_after_epoch is not None:
                self.on_after_epoch(model, pd.DataFrame(self.history))

        return pd.DataFrame(self.history)

    def train_step(self, inputs, labels):
        subbatch_size = int(self.batch_size / self.subdivision)

        for subdiv in range(self.subdivision):
            start, end = subdiv * subbatch_size, (subdiv + 1) * subbatch_size
            sub_inputs, sub_labels = inputs[start:end], labels[start:end]

            sub_inputs.to(self.device)
            sub_labels.to(self.device)

            loss =




    def _train_on_epoch(self, model, optimizer, epoch):
        model.train()
        data_loader = self.data_loaders[0]
        running_loss = 0.0

        optimizer.zero_grad()

        data_loader_len = len(data_loader)
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler(optimizer, epoch, batch_idx, data_loader_len)

            if self.subdivision != 1:
                minibatch_size = int(self.batch_size / self.subdivision)

                batch_idxs = (
                    (start, end)
                    for start, end in zip(
                        range(0, self.batch_size - 1, minibatch_size),
                        range(minibatch_size, self.batch_size + 1, minibatch_size),
                    )
                )

                for start, end in batch_idxs:
                    sub_input, sub_labels = inputs[start:end], labels[start:end]

                    if len(sub_input) != 0:
                        sub_input = sub_input.to(self.device)
                        sub_labels = sub_labels.to(self.device)

                        pred = model(sub_input)
                        loss = self.loss(pred, sub_labels)
                        loss.backward()

                with torch.set_grad_enabled(True):
                    optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item() * inputs.size(0)
            else:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    pred = model(inputs)
                    loss = self.loss(pred, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)
        return epoch_loss

    def _val_on_epoch(self, model, optimizer):
        model.eval()
        data_loader = self.data_loaders[1]
        running_loss = 0.0
        for inputs, labels in data_loader:
            if self.subdivision != 1:
                minibatch_size = int(self.batch_size / self.subdivision)
                batch_idxs = (
                    (start, end)
                    for start, end in zip(
                        range(0, self.batch_size - 1, minibatch_size),
                        range(minibatch_size, self.batch_size + 1, minibatch_size),
                    )
                )
                for start, end in batch_idxs:
                    sub_input, sub_labels = inputs[start:end], labels[start:end]

                    if len(sub_input) != 0:
                        sub_input = sub_input.to(self.device)
                        sub_labels = sub_labels.to(self.device)

                        pred = model(sub_input)
                        loss = self.loss(pred, sub_labels)
                        loss.backward()

                        # pred: minibatch, cls, y, x
                        pred[pred < 0.5] = 0
                        pred[pred >= 0.5] = 1

                        fg_pred = pred[:, 0]
                        bg_pred = pred[:, 1]

                        pred = (bg_pred + (1 - fg_pred)) / 2

                        # TODO
                        tp = (pred == sub_labels).sum()



                with torch.set_grad_enabled(False):
                    optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item() * inputs.size(0)

            else:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(False):
                    pred = model(inputs)
                    loss = self.loss(pred, labels)
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        return epoch_loss
