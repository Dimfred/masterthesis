#!/usr/bin/env python3

import argparse
import logging
import os
from albumentations.augmentations.transforms import CLAHE
from albumentations.core.composition import OneOf

import numpy as np
import pandas as pd
import math

import torch
from torch import optim as optimizers
from torchgeometry import losses
import torchvision as tv

from sklearn.model_selection import KFold

# from tensorboardX import SummaryWriter
# from tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
import torch.nn as nn

import albumentations as A
import cv2 as cv

from dataset import MaskDataset

from nets.MobileNetV2_unet import MobileNetV2_unet

import fastseg
from trainer import Trainer
from adapted_mobilenetv3 import AdaptedMobileNetV3

# dimfred
import utils
from config import config


pad_value = 0
mask_value = 0


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction="none"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target):
        pred = torch.squeeze(pred)
        pred = torch.clip(self.sigmoid(pred), self.eps, 1.0 - self.eps)
        where_true_cls = target == 1

        p_t = torch.where(where_true_cls, pred, 1 - pred)
        alpha = self.alpha * torch.ones_like(pred)
        alpha_t = torch.where(where_true_cls, alpha, 1 - alpha)

        loss = -alpha_t * torch.pow(1 - p_t, self.gamma) * torch.log(p_t)

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = torch.mean(loss, dim=2)
            loss = torch.mean(loss, dim=1)
            loss = loss / loss.shape[0]
            loss = torch.sum(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            raise NotImplementedError(f"Invalid reduction: {self.reduction}")

        return loss


# %%
def get_data_loaders(train_files, val_files, img_size=224):
    pad_size = int(1.1 * config.augment.unet.img_params.resize)
    crop_size = int(config.unet.augment.crop_size * pad_size)

    # print(pad_size, img_size)
    # fmt:off
    train_transform = A.Compose([
        # SafeRotate!
        # rotation
        # A.PadIfNeeded(
        #     min_width=800,
        #     min_height=800,
        #     border_mode=cv.BORDER_CONSTANT,
        #     value=pad_value,
        #     mask_value=mask_value,
        #     always_apply=True
        # ),
        A.PadIfNeeded(
            min_width=pad_size,
            min_height=pad_size,
            border_mode=cv.BORDER_CONSTANT,
            value=pad_value,
            mask_value=mask_value,
            always_apply=True
        ),
        A.Rotate(
            limit=config.unet.augment.rotate,
            border_mode=cv.BORDER_CONSTANT,
            value=pad_value,
            mask_value=mask_value,
            p=0.5
        ),
        A.RandomCrop(
            width=crop_size,
            height=crop_size,
            p=0.5,
        ),
        # A.OneOf(
        #     [
        #         A.ColorJitter(
        #             brightness=config.unet.augment.color_jitter,
        #             contrast=config.unet.augment.color_jitter,
        #             saturation=config.unet.augment.color_jitter,
        #             hue=config.unet.augment.color_jitter,
        #             # p=0.5
        #         ),
        #         A.CLAHE(
        #             # p=0.5
        #         ),
        #     ],
        #     p=0.5
        # ),
        # A.ColorJitter(
        #     brightness=config.unet.augment.color_jitter,
        #     contrast=config.unet.augment.color_jitter,
        #     saturation=config.unet.augment.color_jitter,
        #     hue=config.unet.augment.color_jitter,
        #     p=0.5
        # ),
        # A.CLAHE(
        #     p=0.5
        # ),
        # A.Sharpen(
        #     p=0.5
        # ),
        # A.Blur(
        #     blur_limit=config.unet.augment.blur,
        #     p=0.5
        # ),
        A.Resize(
            width=img_size,
            height=img_size,
            always_apply=True
        ),
    ])

    valid_transform = A.Compose([
        A.PadIfNeeded(
            min_width=config.augment.unet.img_params.resize,
            min_height=config.augment.unet.img_params.resize,
            border_mode=cv.BORDER_CONSTANT,
            value=pad_value,
            mask_value=mask_value,
            always_apply=True
        ),
        A.Resize(
            width=img_size,
            height=img_size,
            always_apply=True
        ),
    ])
    # fmt:on

    train_loader = DataLoader(
        MaskDataset(train_files, train_transform, channels=config.unet.channels),
        batch_size=config.unet.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.unet.n_workers,
        prefetch_factor=3,
    )
    valid_loader = DataLoader(
        MaskDataset(val_files, valid_transform, channels=config.unet.channels),
        batch_size=config.unet.valid_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.unet.n_workers,
    )

    return train_loader, valid_loader


# TODO print training params at the beginning
def main():
    import sys

    run_ = sys.argv[1]
    for run in (int(run_),):
        # for run in (0, 1, 2):
        seed = config.train.seeds[run]
        utils.seed_all(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")

        train_files = utils.list_imgs(config.train_out_dir)
        val_files = utils.list_imgs(config.valid_out_dir)
        data_loaders = get_data_loaders(train_files, val_files, config.unet.input_size)

        ##########
        ## LOSS ##
        ##########
        loss_type = "focal"
        loss = losses.focal.FocalLoss(
            config.unet.focal_alpha,
            config.unet.focal_gamma,
            config.unet.focal_reduction,
        )

        # loss_type = "dice"
        # loss = losses.dice.DiceLoss()

        # loss_type = "binfocal"
        # loss = BinaryFocalLoss(
        #     config.unet.focal_alpha,
        #     config.unet.focal_gamma,
        #     config.unet.focal_reduction,
        # )

        ##########
        ### V2 ###
        ##########

        if config.unet.architecture == "v2":
            model = MobileNetV2_unet(
                n_classes=config.unet.n_classes if loss_type == "focal" else 1,
                input_size=config.unet.input_size,
                channels=config.unet.channels,
                pretrained=config.unet.pretrained_path,
                width_multiplier=config.unet.width_multiplier,
            )
            if (
                config.unet.checkpoint_path is not None
                and config.unet.pretrained_path is None
            ):
                print("Preloaded checkpoing path.")
                model.load_state_dict(torch.load(str(config.unet.checkpoint_path)))

        ##########
        ### V3 ###
        ##########
        if config.unet.architecture == "v3":
            # model = fastseg.MobileV3Large(num_classes=2)
            model = fastseg.MobileV3Large(num_classes=2).from_pretrained()

        if config.unet.architecture == "unet":

            class DeeplabV3Resnet50(nn.Module):
                def __init__(self, *args, **kwargs):
                    super(DeeplabV3Resnet50, self).__init__()
                    self.model = tv.models.segmentation.deeplabv3_resnet50(
                        *args, **kwargs
                    )

                def forward(self, x):
                    x = self.model(x)["out"]
                    return x

            model = DeeplabV3Resnet50(num_classes=2)

        model.to(device)

        ###############
        ## OPTIMIZER ##
        ###############
        optimizer = optimizers.Adam(
            model.parameters(),
            lr=config.unet.lr,
            betas=config.unet.betas,
            weight_decay=config.unet.decay,
            amsgrad=config.unet.amsgrad,
            # nesterov=config.unet.nesterov,
        )

        def lr_scheduler(optimizer, step):
            lr = config.unet.lr

            if step in config.unet.lr_decay_fixed:
                lr = lr / 2
                print("Halfing LR:", lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            return lr

        # optimizer = optimizers.SGD(
        #     model.parameters(), lr=config.unet.lr, momentum=config.unet.momentum
        # )
        # def lr_scheduler(optimizer, step):
        #     lr = config.unet.lr

        #     # if step < config.unet.burn_in:
        #     #     multiplier = (step / config.unet.burn_in) ** 4
        #     #     lr = lr * multiplier

        #     # for param_group in optimizer.param_groups:
        #     #     param_group["lr"] = lr
        #     # print(lr)

        #     return lr

        experiment = utils.UnetExperiment(
            config.unet.experiment_dir,
            config.unet.experiment_name,
            config.unet.experiment_param,
            run,
        )

        trainer = Trainer(
            data_loaders=data_loaders,
            loss=loss,
            batch_size=config.unet.batch_size,
            subdivision=config.unet.subdivision,
            valid_batch_size=config.unet.valid_batch_size,
            valid_subdivision=config.unet.valid_subdivision,
            lr_scheduler=lr_scheduler,
            device=device,
            experiment=experiment,
            loss_type=loss_type,
        )
        trainer.train(model, optimizer)


if __name__ == "__main__":
    main()

# %%
