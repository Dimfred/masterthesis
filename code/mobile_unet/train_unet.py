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


EXPERIMENT = "unet"
OUT_DIR = "outputs/{}".format(EXPERIMENT)

# dimfred
import utils
from config import config


# %%
def get_data_loaders(train_files, val_files, img_size=224):
    train_transform = A.Compose(
        [
            A.RandomScale(scale_limit=(0.7, 1.3), interpolation=cv.INTER_CUBIC, p=0.3),
            A.Resize(
                img_size, img_size, interpolation=cv.INTER_CUBIC, always_apply=True
            ),
            # rotation
            A.RandomRotate90(p=1.0),
            A.Rotate(30, border_mode=cv.BORDER_CONSTANT, p=0.3),
            A.HorizontalFlip(p=0.5),
            # pixel augmentation
            A.RandomBrightnessContrast(),
            A.RandomGamma((90, 110), p=0.3),
            A.CLAHE(),
            A.GaussianBlur((3, 3), sigma_limit=1.2, p=0.3),
            # A.HueSaturationValue(
            #     hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3
            # ),
            # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            # removal
            A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.3),
            A.RandomCrop(
                img_size,
                img_size,
                p=0.3,
            )
            # TODO
            # A.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
            # TODO ???
            # ToTensorV2(),
        ]
    )

    valid_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            # A.GaussianBlur(limit=(5, 5), sigma_limit=1.2, always_apply=True),
        ]
    )

    train_loader = DataLoader(
        MaskDataset(train_files, train_transform),
        batch_size=config.unet.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.unet.n_workers,
    )
    valid_loader = DataLoader(
        MaskDataset(val_files, valid_transform),
        batch_size=config.unet.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.unet.n_workers,
    )

    return train_loader, valid_loader


def save_best_model(model, df_hist):
    if df_hist["val_loss"].tail(1).iloc[0] <= df_hist["val_loss"].min():
        torch.save(model.state_dict(), "{}/best.pth".format(OUT_DIR))


def write_on_board(writer, df_hist):
    row = df_hist.tail(1).iloc[0]

    writer.add_scalars(
        "{}/loss".format(EXPERIMENT),
        {
            "train": row.train_loss,
            "val": row.val_loss,
            "lr": row.lr,
        },
        row.epoch,
    )


def log_hist(df_hist):
    last = df_hist.tail(1)
    best = df_hist.sort_values("val_loss").head(1)
    summary = pd.concat((last, best)).reset_index(drop=True)
    summary["name"] = ["Last", "Best"]
    logger.debug(summary[["name", "epoch", "train_loss", "val_loss"]])
    logger.debug("")


# TODO print training params at the beginning
def run_training(img_size, pretrained):
    np.random.seed(config.unet.random_state)
    torch.manual_seed(config.unet.random_state)
    torch.backends.cudnn.deterministic = True

    writer = SummaryWriter(flush_secs=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def on_after_epoch(m, df_hist):
        # save_best_model(n, m, df_hist)
        save_best_model(m, df_hist)
        write_on_board(writer, df_hist)
        log_hist(df_hist)

    train_files = utils.list_imgs(config.train_out_dir)
    val_files = utils.list_imgs(config.valid_out_dir)
    data_loaders = get_data_loaders(train_files, val_files, config.unet.input_size)

    model = MobileNetV2_unet(
        n_classes=config.unet.n_classes,
        input_size=config.unet.input_size,
        channels=config.unet.channels,
        pretrained=config.unet.pretrained_path,
    )
    if config.unet.checkpoint_path is not None and config.unet.pretrained_path is None:
        model.load_state_dict(torch.load(str(config.unet.checkpoint_path)))
    model.to(device)

    ##########
    ## LOSS ##
    ##########
    # loss = losses.dice.DiceLoss()
    loss = losses.focal.FocalLoss(
        config.unet.focal_alpha,
        config.unet.focal_gamma,
        config.unet.focal_reduction,
    )
    # TODO tversky

    ###############
    ## OPTIMIZER ##
    ###############
    optimizer = optimizers.Adam(
        model.parameters(),
        lr=config.unet.lr,
        betas=config.unet.betas,
        weight_decay=config.unet.decay,
        amsgrad=config.unet.amsgrad,
    )
    # TODO sgd

    trainer = Trainer(
        data_loaders,
        loss,
        device,
        batch_size=config.unet.batch_size,
        subdivision=config.unet.subdivision,
        on_after_epoch=on_after_epoch,
        lr_scheduler=lr_scheduler,
    )
    hist = trainer.train(model, optimizer, num_epochs=config.unet.n_epochs)

    hist.to_csv("{}/{}-hist.csv".format(OUT_DIR, 0), index=False)
    writer.close()


def lr_scheduler(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]["lr"]

    warmup_epoch = config.unet.lr_burn_in

    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = config.unet.n_epochs * num_iter

    if config.unet.lr_decay == "step":
        # TODO args
        lr = args.lr * (
            args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter))
        )
    elif config.unet.lr_decay == "cos":
        lr = (
            config.unet.lr
            * (
                1
                + math.cos(
                    math.pi * (current_iter - warmup_iter) / (max_iter - warmup_iter)
                )
            )
            / 2
        )
    elif config.unet.lr_decay == "fixed":
        if config.unet.lr_decay_fixed and epoch == config.unet.lr_decay_fixed[0]:
            lr = lr / 10
            config.unet.lr_decay_fixed.pop(0)
    elif config.unet.lr_decay == "linear":
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == "schedule":
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError("Unknown lr mode {}".format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = config.unet.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


if __name__ == "__main__":
    if not config.unet.output_dir.exists():
        os.makedirs(config.unet.output_dir)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(
            logging.FileHandler(filename="outputs/{}.log".format(EXPERIMENT))
        )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="image size",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        help="path of pre trained weight",
    )
    args, _ = parser.parse_known_args()
    # print(args)
    run_training(**vars(args))

# %%
