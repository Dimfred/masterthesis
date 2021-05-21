from glob import glob

import cv2 as cv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# import torchvision

# from config import IMG_DIR
from pathlib import Path
import os
import albumentations as A
import time
from concurrent.futures import ThreadPoolExecutor

# from torchvision.transforms.transforms import ToTensor

import utils


def _img_to_mask(img_file: Path) -> Path:
    # return DATA_LFW_DIR / f"raw/masks/{img_file.stem}.ppm"
    path, ext = os.path.splitext(img_file)
    return f"{path}.npy"


def get_img_files(path: Path) -> np.ndarray:
    img_files = [*path.glob("**/*.jpg", *path.glob("**/*.png"))]
    return np.array(img_files)


class MaskDataset(Dataset):
    def __init__(self, img_files, transform, mask_transform=None):
        self.img_files = [str(f) for f in img_files]
        self.mask_files = [_img_to_mask(f) for f in img_files]
        self.transform = transform

        if mask_transform is None:
            self.mask_transform = transform
        else:
            self.mask_transform = mask_transform

        print("Preloading dataset...")
        tstart = time.perf_counter()
        self.data = []

        def load(img_path, mask_path):
            img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            mask = np.load(mask_path)
            self.data.append((img, mask))

        with ThreadPoolExecutor(max_workers=32) as executor:
            for img_path, mask_path in zip(self.img_files, self.mask_files):
                executor.submit(load, img_path, mask_path)

        print("Took:", f"{time.perf_counter() - tstart:.3f}s")

    def __getitem__(self, idx):
        img_, mask_ = self.data[idx]
        img, mask = img_.copy(), mask_.copy()

        orig = img.copy()

        augmented = self.transform(image=img, mask=mask)
        img = np.array(augmented["image"]).astype(np.float32)
        mask = np.array(augmented["mask"]).astype(np.int64)

        if utils.isme():
            simg = np.uint8(img)
            smask = np.expand_dims(np.uint8(mask * 255), axis=2)
            simg_mask = np.uint8(mask * img)
            utils.show(orig, simg, smask, simg_mask)
            pass

        # utils.show(img)

        # grayscale
        # img = np.expand_dims(img, axis=2)

        # grayscale and rgb pretrained
        img = np.repeat(img[..., np.newaxis], 3, -1)

        img = img / 255.0
        img = img.transpose((2, 0, 1))

        return img, mask

    def __len__(self):
        return len(self.img_files)


if __name__ == "__main__":
    pass
    #
    # mask = cv.imread('{}/masks/Aaron_Peirsol_0001.ppm'.format(IMG_DIR))
    # mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    # mask = mask[:, :, 0]
    # print(mask.shape)
    # plt.imshow(mask)
    # plt.show()
