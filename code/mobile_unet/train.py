#!/usr/bin/env python3

import argparse
import logging
import os

import numpy as np
import pandas as pd
import math

import torch
from torch import optim as optimizers
from torchgeometry import losses

from sklearn.model_selection import KFold

# from tensorboardX import SummaryWriter
# from tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

import albumentations as A
import cv2 as cv

from dataset import MaskDataset

from nets.MobileNetV2_unet import MobileNetV2_unet
from trainer import Trainer

# dimfred
import utils
from config import config


# %%
def get_data_loaders(train_files, val_files, img_size=224):
    # fmt:off
    train_transform = A.Compose([
        # rotation
        # A.PadIfNeeded(
        #     min_width=800,
        #     min_height=800,
        #     border_mode=cv.BORDER_CONSTANT,
        #     value=0,
        #     mask_value=0,
        #     always_apply=True
        # ),
        # A.RandomScale(
        #     scale_limit=config.unet.augment.random_scale,
        #     interpolation=cv.INTER_CUBIC,
        #     p=0.5
        # ),
        # A.Rotate(
        #     limit=config.unet.augment.rotate,
        #     border_mode=cv.BORDER_CONSTANT,
        #     value=0,
        #     mask_value=0,
        #     p=0.5
        # ),
        # A.RandomCrop(
        #     width=int(config.unet.augment.crop_size * img_size),
        #     height=int(config.unet.augment.crop_size * img_size),
        #     p=0.5,
        # ),
        # A.ColorJitter(
        #     brightness=config.unet.augment.color_jitter,
        #     contrast=config.unet.augment.color_jitter,
        #     saturation=config.unet.augment.color_jitter,
        #     hue=config.unet.augment.color_jitter,
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
        # A.PadIfNeeded(
        #     min_width=config.augment.unet.img_params.resize,
        #     min_height=config.augment.unet.img_params.resize,
        #     border_mode=cv.BORDER_CONSTANT,
        #     value=0,
        #     mask_value=0,
        #     always_apply=True
        # ),
        A.Resize(
            width=img_size,
            height=img_size,
            always_apply=True
        ),
    ])
    # fmt:on

    train_loader = DataLoader(
        MaskDataset(train_files, train_transform),
        batch_size=config.unet.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.unet.n_workers,
    )
    valid_loader = DataLoader(
        MaskDataset(val_files, valid_transform),
        batch_size=config.unet.valid_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.unet.n_workers,
    )

    return train_loader, valid_loader


def lr_scheduler(optimizer, step):
    lr = config.unet.lr

    if step < config.unet.burn_in:
        multiplier = (step / config.unet.burn_in) ** 4
        lr = lr * multiplier

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


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

        model = MobileNetV2_unet(
            n_classes=config.unet.n_classes,
            input_size=config.unet.input_size,
            channels=config.unet.channels,
            pretrained=config.unet.pretrained_path,
        )
        if (
            config.unet.checkpoint_path is not None
            and config.unet.pretrained_path is None
        ):
            model.load_state_dict(torch.load(str(config.unet.checkpoint_path)))
        model.to(device)

        ###############
        ## OPTIMIZER ##
        ###############
        # optimizer = optimizers.Adam(
        #     model.parameters(),
        #     lr=config.unet.lr,
        #     betas=config.unet.betas,
        #     weight_decay=config.unet.decay,
        #     amsgrad=config.unet.amsgrad,
        # )

        optimizer = optimizers.SGD(
            model.parameters(), lr=config.unet.lr, momentum=config.unet.momentum
        )

        ##########
        ## LOSS ##
        ##########
        loss = losses.focal.FocalLoss(
            config.unet.focal_alpha,
            config.unet.focal_gamma,
            config.unet.focal_reduction,
        )

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
            valid_batch_size=config.unet.valid_batch_size,
            subdivision=config.unet.subdivision,
            lr_scheduler=lr_scheduler,
            device=device,
            experiment=experiment,
        )
        trainer.train(model, optimizer)


if __name__ == "__main__":
    main()

# %%
