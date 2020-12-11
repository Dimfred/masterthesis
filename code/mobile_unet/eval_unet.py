#!/usr/bin/env python3

import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from dataset import MaskDataset #get_img_files, get_img_files_eval
from nets.MobileNetV2_unet import MobileNetV2_unet
import albumentations as A

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
    val_transform = A.Compose(
        [
            A.Resize(config.unet.input_size, config.unet.input_size),
            # A.GaussianBlur((5, 5), sigma_limit=1.2, always_apply=True),
        ]
    )

    val_loader = DataLoader(
        MaskDataset(val_files, val_transform),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        # num_workers=1,
    )

    return val_loader


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
    val_files = utils.list_imgs(config.valid_out_dir)
    data_loader = get_data_loaders(val_files)

    model = MobileNetV2_unet(
        mode="eval",
        n_classes=config.unet.n_classes,
        input_size=config.unet.input_size,
        channels=config.unet.channels,
        pretrained=None
    )
    # CPU version
    # model.load_state_dict(torch.load('{}/{}-best.pth'.format(OUT_DIR, n), map_location="cpu"))
    # GPU version
    loaded = torch.load("weights/best.pth")
    model.load_state_dict(loaded)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            for i, l, o in zip(inputs, labels, outputs):
                # move channels back
                i = i.cpu().numpy().transpose((1, 2, 0)) * 255
                # l = l.cpu().numpy().reshape(*img_size)
                l = l.cpu().numpy() * 255
                o = o.cpu().numpy() * 255
                print(o.shape)
                # o = o.cpu().numpy().reshape(int(IMG_SIZE / 2), int(IMG_SIZE / 2)) * 255
                # o = o.cpu().numpy().reshape(int(IMG_SIZE), int(IMG_SIZE)) #* 255

                # i = cv2.resize(i.astype(np.uint8), img_size)
                # l = cv2.resize(l.astype(np.uint8), img_size)
                # o = cv2.resize(o.astype(np.uint8), img_size)

                i = np.uint8(i)
                l = np.uint8(l)
                o = np.uint8(o)

                utils.show(i, l, o[..., np.newaxis])

                # plt.subplot(131)
                # plt.imshow(i)
                # plt.subplot(132)
                # plt.imshow(l)
                # plt.subplot(133)
                # plt.imshow(o)
                # plt.show()
                n_shown += 1
                # if n_shown > 10:
                #     return


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
