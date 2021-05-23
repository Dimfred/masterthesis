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

from dataset import MaskDataset #get_img_files, get_img_files_eval
from nets.MobileNetV2_unet import MobileNetV2_unet
import albumentations as A
import numba as nb
from tabulate import tabulate

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
def get_data_loaders(val_files):
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

    val_loader = DataLoader(
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
            if target_val == 1:
                if prediction_val == 1:
                    tps += 1
                else:
                    fns += 1
            else:
                if prediction_val == 1:
                    fps += 1

    return tps, fps, fns

def ffloat(f):
    return "{:.5f}".format(f * 100)

def evaluate():
    n_shown = 0

    # image_files = ["imgs/path"]
    # image_files = get_img_files_eval()
    # kf = KFold(n_splits=N_CV, random_state=RANDOM_STATE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # for n, (train_idx, val_idx) in enumerate(kf.split(image_files)):
    # for n, img_path in enumerate(image_files):
        # val_files = image_files[val_idx]
        # val_files = image_files[n]
    # val_files = utils.list_imgs(config.valid_out_dir)
    # data_loader = get_data_loaders(val_files)


    model = MobileNetV2_unet(
        mode="eval",
        n_classes=config.unet.n_classes,
        input_size=config.unet.test_input_size,
        channels=config.unet.channels,
        pretrained=None,
        scale=config.unet.scale,
    )
    # CPU version
    # model.load_state_dict(torch.load('{}/{}-best.pth'.format(OUT_DIR, n), map_location="cpu"))
    # GPU version
    # unable to load it anymore

    # loaded = torch.load("weights/non_transfer_best.pth")
    # loaded = torch.load("weights/best.pth")
    loaded = torch.load("experiments_unet/test/test/run0/best.pth")
    # loaded = torch.load("weights/best_safe_FUCKING_KEEP_IT.pth")

    model.load_state_dict(loaded)
    model.to(device)
    model.eval()

    import sys
    dir_ = sys.argv[1]
    show = len(sys.argv) > 2

    if dir_ == "test":
        test_files = utils.list_imgs(config.test_out_dir)
    elif dir_ == "valid":
        test_files = utils.list_imgs(config.valid_out_dir)
    elif dir_ == "train":
        test_files = utils.list_imgs(config.train_out_dir)
    data_loader = get_data_loaders(test_files)

    n_shown = 0
    ious = []

    conf_thresh = 0.6

    tps, fps, fns = 0, 0, 0
    with torch.no_grad():

        for idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            for i, l, o in zip(inputs, labels, outputs):
                # move channels back
                i = i.cpu().numpy().transpose((1, 2, 0)) * 255
                # l = l.cpu().numpy().reshape(*img_size)
                o = o.cpu().numpy()
                l = l.cpu().numpy()
                # o = o.cpu().numpy() * 255
                # print(o.shape)

                # prepare pred
                bg, fg = o[0], o[1]
                o = (fg + 1 - bg) / 2
                o[o >= conf_thresh] = 1
                o[o < conf_thresh] = 0

                o = np.uint8(o)
                l = np.uint8(l)

                # metrics
                iou = calc_iou(l, o)
                ious.append(iou)
                print(iou)


                tps_, fps_, fns_ = tps_fps_fns(l, o)
                tps += tps_
                fps += fps_
                fns += fns_

                # DEBUG
                o = o * 255
                l = l * 255
                i = np.uint8(i)
                l = np.uint8(l)
                o = np.uint8(o)

                if show and iou < 0.8:
                    utils.show(i, l, o[..., np.newaxis]) #, i * np.logical_not(o[..., np.newaxis]))




    print("All:", ious)

    miou = np.array(ious).mean()
    recall = calc_recall(tps, fns)
    precision = calc_precision(tps, fps)
    f1 = calc_f1(precision, recall)

    pretty = [["mIoU", "Precision", "Recall", "F1"]]
    pretty += [[ffloat(miou), ffloat(precision), ffloat(recall), ffloat(f1)]]
    print(tabulate(pretty))

def calc_iou(target, prediction):
    # print(target.shape)
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection, axis=(0, 1)) / np.sum(union, axis=(0, 1))

    return iou_score.mean()

if __name__ == "__main__":
    # if not os.path.exists(OUT_DIR):
    #     os.makedirs(OUT_DIR)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(
            logging.FileHandler(filename="outputs/{}.log".format(EXPERIMENT))
        )

    evaluate()
