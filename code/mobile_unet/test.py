#!/usr/bin/env python3.8

import logging
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from dataset import MaskDataset  # get_img_files, get_img_files_eval
from nets.MobileNetV2_unet import MobileNetV2_unet
import albumentations as A
import numba as nb
from tabulate import tabulate
import click

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)

# %%


EXPERIMENT = "unet"
# EXPERIMENT = '10000_day'
# OUT_DIR = 'outputs/{}/first_shot'.format(EXPERIMENT)
# OUT_DIR = 'outputs/{}'.format(EXPERIMENT)
# OUT_DIR = "outputs/UNET_224_weights_100000_days"

N = 0

import utils
from config import config

# %%
# fmt: off
val_transform = A.Compose(
    [
        # A.Blur(always_apply=True),
        A.PadIfNeeded(
            min_width=config.augment.unet.img_params.resize,
            min_height=config.augment.unet.img_params.resize,
            border_mode=cv.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            always_apply=True
        ),
        A.Resize(
            width=config.unet.test_input_size,
            height=config.unet.test_input_size,
            always_apply=True
        ),
    ]
)
# fmt: on


def get_data_loaders(val_files):
    val_loader = DataLoader(
        # MaskDataset(val_files, val_transform),
        MaskDataset(val_files, val_transform),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        # num_workers=1,
    )

    return val_loader


def calc_recall(tp, fn):
    return tp / (tp + fn)


def calc_precision(tp, fp):
    return tp / (tp + fp)


def calc_f1(precision, recall):
    return 2 * precision * recall / (precision + recall)


@nb.njit
def tps_fps_fns(target, prediction):
    tps, fps, fns = 0, 0, 0
    # for target_, prediction_ in zip(target, prediction):
    for target_row, prediction_row in zip(target, prediction):
        for target_val, prediction_val in zip(target_row, prediction_row):
            if target_val == 255:
                if prediction_val == 255:
                    tps += 1
                else:
                    fns += 1
            else:
                if prediction_val == 255:
                    fps += 1

    return tps, fps, fns


def ffloat(f):
    return "{:.5f}".format(f * 100)

def get_exp_weights():
    exp_dir = config.unet.experiment_dir
    ####################################################################################
    ## LR
    ####################################################################################
    exp_base = exp_dir / "lr"
    exp = lambda val, run: exp_base / f"lr_{val}/run{run}/best.pth"

    weights = []
    for val in (0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001):
        for run in (0, 1, 2):
            weights.append(exp(val, run))

    return weights




@click.command()
@click.argument("dataset")
@click.option("-s", "--score_thresh", type=click.types.FLOAT, default=0.5)
@click.option("--exp", is_flag=True, default=False)
@click.option("--tta", is_flag=True, default=False)
@click.option("--show", is_flag=True, default=False)
def main(dataset, score_thresh, exp, tta, show):
    exp = True if exp else False
    tta = True if tta else False
    show = True if show else False

    if dataset == "test":
        test_files = utils.list_imgs(config.test_out_dir)
    elif dataset == "valid":
        test_files = utils.list_imgs(config.valid_out_dir)
    elif dataset == "train":
        test_files = utils.list_imgs(config.train_out_dir)
    data_loader = get_data_loaders(test_files)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2_unet(
        mode="eval",
        n_classes=config.unet.n_classes,
        input_size=config.unet.test_input_size,
        channels=config.unet.channels,
        pretrained=None,
        scale=config.unet.scale,
    )

    # loaded = torch.load("weights/non_transfer_best.pth")
    # loaded = torch.load("weights/best.pth")
    # loaded = torch.load("experiments_unet/test/test/run0/best.pth")
    # loaded = torch.load("weights/best_safe_FUCKING_KEEP_IT.pth")

    if exp:
        weights = get_exp_weights()
    else:
        weights = ["weights/best_78miou@608_trained_with_448.pth"]



    for weight_path in weights:
        print(weight_path)

        loaded = torch.load(weight_path)
        model.load_state_dict(loaded)
        model.to(device)
        model.eval()

        ious = []

        tps, fps, fns = 0, 0, 0
        with torch.no_grad():

            for img_path in test_files:
                # print(img_path)

                label_path = utils.segmentation_label_from_img(img_path)
                label = np.load(str(label_path))
                # label = utils.resize_max_axis(label, config.unet.test_input_size)

                img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)

                pred, label = model.predict(
                    img, label=label, score_thresh=score_thresh, tta=tta, debug=show
                )

                # metrics
                iou = calc_iou(label, pred)
                ious.append(iou)

                tps_, fps_, fns_ = tps_fps_fns(label, pred)
                tps += tps_
                fps += fps_
                fns += fns_

                # DEBUG
                if show:
                    utils.show(img, label, pred)

        # print("All:", ious)

        miou = np.array(ious).mean()
        recall = calc_recall(tps, fns)
        precision = calc_precision(tps, fps)
        f1 = calc_f1(precision, recall)

        pretty = [["ds", "input", "TTA", "mIoU", "Precision", "Recall", "F1"]]
        pretty += [
            [
                dataset,
                config.unet.test_input_size,
                tta,
                ffloat(miou),
                ffloat(precision),
                ffloat(recall),
                ffloat(f1),
            ]
        ]
        print(tabulate(pretty))


def calc_iou(target, prediction):
    # print(target.shape)
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection, axis=(0, 1)) / np.sum(union, axis=(0, 1))

    return iou_score.mean()


if __name__ == "__main__":
    main()
