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

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx])
        img = np.array(img)

        mask = np.load(self.mask_files[idx])

        augmented = self.transform(image=img, mask=mask)

        mask = np.array(augmented["mask"]).astype(np.int64)
        img = np.array(augmented["image"]).astype(np.float32)

        if utils.isme():
            utils.show(
               cv.cvtColor(np.uint8(img), cv.COLOR_RGB2BGR)
               * np.uint8(mask)[..., np.newaxis]
            )
            pass

        img = img.transpose((2, 0, 1)) / 255.0

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
