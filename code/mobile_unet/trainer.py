import pandas as pd
import numpy as np
import torch


class Trainer:
    def __init__(
        self,
        data_loaders,
        criterion,
        device,
        batch_size=32,
        subdivision=1,
        on_after_epoch=None,
        lr_scheduler=None,
    ):
        self.data_loaders = data_loaders
        self.criterion = criterion
        self.device = device
        self.history = []
        self.on_after_epoch = on_after_epoch

        self.batch_size = batch_size
        assert self.batch_size % 2 == 0

        self.subdivision = subdivision
        assert subdivision > 0

        self.lr_scheduler = lr_scheduler

    def train(self, model, optimizer, num_epochs):
        for epoch in range(num_epochs):
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
                        loss = self.criterion(pred, sub_labels)
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
                    loss = self.criterion(pred, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)
        return epoch_loss

        # inputs = inputs.to(self.device)
        # # labels = one_hot()
        # labels = labels.to(self.device)
        # pred = model(inputs)
        # loss = self.criterion(pred, labels)
        # loss.backward()

        # if (self.batch_counter + 1) % self.subdivision == 0:

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
                        loss = self.criterion(pred, sub_labels)
                        loss.backward()

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
                    loss = self.criterion(pred, labels)
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        return epoch_loss
