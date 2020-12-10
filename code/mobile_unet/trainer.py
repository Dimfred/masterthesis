import pandas as pd
import numpy as np
import torch


class Trainer:
    def __init__(
        self,
        data_loaders,
        criterion,
        device,
        subdivision=1,
        on_after_epoch=None,
        lr_scheduler=None,
    ):
        self.data_loaders = data_loaders
        self.criterion = criterion
        self.device = device
        self.history = []
        self.on_after_epoch = on_after_epoch

        self.subdivision = subdivision
        assert subdivision > 0

        self.batch_counter = 0
        self.lr_scheduler = lr_scheduler

    def train(self, model, optimizer, num_epochs):
        for epoch in range(num_epochs):
            train_epoch_loss = self._train_on_epoch(model, optimizer, epoch)
            val_epoch_loss = self._val_on_epoch(model, optimizer)

            hist = {
                "epoch": epoch,
                "train_loss": train_epoch_loss,
                "val_loss": val_epoch_loss,
            }
            self.history.append(hist)

            if self.on_after_epoch is not None:
                self.on_after_epoch(model, pd.DataFrame(self.history))

        return pd.DataFrame(self.history)

    def _train_on_epoch(self, model, optimizer, epoch):
        model.train()
        data_loader = self.data_loaders[0]
        running_loss = 0.0

        optimizer.zero_grad()

        data_loader_len = len(data_loader)
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler(optimizer, epoch, batch_idx, data_loader_len)
                print("Learning_rate:", lr)

            self.batch_counter += 1

            inputs = inputs.to(self.device)
            # labels = one_hot()
            labels = labels.to(self.device)

            pred = model(inputs)
            loss = self.criterion(pred, labels)
            loss.backward()

            if (self.batch_counter + 1) % self.subdivision == 0:
                with torch.set_grad_enabled(True):
                    optimizer.step()

                optimizer.zero_grad()

            running_loss += loss.item() * inputs.size(0)

            # with torch.set_grad_enabled(True):
            # outputs = outputs.to("cpu")
            # labels = labels.to("cpu")
            # outputs = torch.cat([outputs, 1 - outputs], 1)
            # print(outputs.shape)
            # loss = self.criterion(pred, labels)
            # loss.backward()
            # optimizer.step()

            # TODO why the fuck does the model not output the same size???
            # labels = torch.nn.functional.interpolate(
            #     labels, scale_factor=0.5, mode="linear", align_corners=False
            # )
            # outputs = torch.nn.functional.interpolate(
            #     outputs, scale_factor=2, mode="bilinear", align_corners=False
            # )
            # print("outputs.shape\n{}".format(outputs.shape))

            # running_loss += loss.item() #* inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        return epoch_loss

    def _val_on_epoch(self, model, optimizer):
        model.eval()
        data_loader = self.data_loaders[1]
        running_loss = 0.0

        for inputs, labels in data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        return epoch_loss
