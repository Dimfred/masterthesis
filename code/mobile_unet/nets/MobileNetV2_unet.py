import logging
import math
import sys

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import cv2 as cv
import numpy as np

from .MobileNetV2 import MobileNetV2, InvertedResidual
from .MobileNetV2 import conv_1x1_bn

from config import config
import utils

import albumentations as A

# logging.basicConfig(level=logging.DEBUG)

# paper implementation
class MobileNetV2_unet(nn.Module):
    def __init__(
        self,
        n_classes=1000,
        input_size=224,
        channels=3,
        pretrained="weights/mobilenet_v2.pth.tar",
        width_multiplier=1.0,
        mode="train",
        scale=False,
        upsampling="transpose",
    ):
        super(MobileNetV2_unet, self).__init__()

        self.mode = mode
        self.input_size = input_size
        self.scale = scale

        self.backbone = MobileNetV2(
            n_classes=1000,
            input_size=input_size,
            channels=channels,
            width_mult=width_multiplier,
        )

        # input_channel, last_channel
        # self.dense0 = conv_1x1_bn(320, 1280)

        if upsampling == "transpose":
            if width_multiplier == 1.0:
                self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
                self.invres1 = InvertedResidual(192, 96, 1, 6)

                self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
                self.invres2 = InvertedResidual(64, 32, 1, 6)

                self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
                self.invres3 = InvertedResidual(48, 24, 1, 6)

                self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)

                if not scale:
                    self.invres4 = InvertedResidual(32, 16, 1, 6)
                    self.dconv5 = nn.ConvTranspose2d(
                        16, n_classes, 4, padding=1, stride=2
                    )
                else:
                    self.invres4 = InvertedResidual(32, n_classes, 1, 6)
                    self.dconv5 = nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=False
                    )

            elif width_multiplier == 1.4:
                self.dconv1 = nn.ConvTranspose2d(1792, 136, 4, padding=1, stride=2)
                self.invres1 = InvertedResidual(272, 136, 1, 6)

                self.dconv2 = nn.ConvTranspose2d(136, 48, 4, padding=1, stride=2)
                self.invres2 = InvertedResidual(96, 48, 1, 6)

                self.dconv3 = nn.ConvTranspose2d(48, 32, 4, padding=1, stride=2)
                self.invres3 = InvertedResidual(64, 32, 1, 6)

                self.dconv4 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
                self.invres4 = InvertedResidual(48, 24, 1, 6)

                self.dconv5 = nn.ConvTranspose2d(24, n_classes, 4, padding=1, stride=2)

        elif upsampling == "bilinear":
            if width_multiplier == 1.0:

                def Upsample():
                    return nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=False
                    )

                # def Upsample(inp, outp):
                #     return nn.Sequential(
                #         InvertedResidual(inp, outp, 1, 6),
                #     )

                self.dconv1 = nn.Sequential(
                    # TODO test order
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    InvertedResidual(1280, 96, 1, 6),
                )
                self.invres1 = InvertedResidual(192, 32, 1, 6)

                self.dconv2 = Upsample()
                self.invres2 = InvertedResidual(64, 24, 1, 6)

                self.dconv3 = Upsample()
                self.invres3 = InvertedResidual(48, 16, 1, 6)

                self.dconv4 = Upsample()
                self.invres4 = InvertedResidual(32, n_classes, 1, 6)

                self.dconv5 = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=False
                )

        # self.invres5 = None

        # dimfred
        # self.dconv5 = nn.ConvTranspose2d(16, 8, 4, padding=1, stride=2)
        # self.invres5 = InvertedResidual(8, n_classes, 1, 6)

        # self.sigmoid = nn.Sigmoid()
        # self.conv_last = nn.Conv2d(8, 3, 1)

        # self.softmax = nn.Softmax(dim=1)

        # original
        # self.conv_last = nn.Conv2d(16, 3, 1)

        # for > 2 classes
        # self.conv_score = nn.Conv2d(3, n_class, 1)

        # self.conv_score = nn.Conv2d(3, 1, 1)

        self._init_weights()

        if pretrained is not None:
            print("Loading pretrained weights...")
            # self.backbone.load_state_dict(torch.load(pre_trained, map_location="cpu"))
            self.backbone.load_state_dict(torch.load(pretrained))
            print("Done loading weights.")

    def forward(self, x, *args, **kwargs):
        # print((x.shape, "input"))
        for n in range(0, 2):
            x = self.backbone.features[n](x)

        x1 = x
        # print((x1.shape, "x1"))

        for n in range(2, 4):
            x = self.backbone.features[n](x)
        x2 = x
        # print((x2.shape, "x2"))

        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x
        # print((x3.shape, "x3"))

        for n in range(7, 14):
            x = self.backbone.features[n](x)
        x4 = x
        # print((x4.shape, "x4"))

        # TODO 1x1 layer removed hence 18 instead of 19
        for n in range(14, 18):
            x = self.backbone.features[n](x)

        x5 = self.backbone.conv(x)
        # print((x5.shape, "x5"))

        dc1 = self.dconv1(x5)
        up1 = torch.cat([x4, dc1], dim=1)
        up1 = self.invres1(up1)
        # print((dc1.shape, "dc1"))
        # print((up1.shape, "up1"))

        dc2 = self.dconv2(up1)
        up2 = torch.cat([x3, dc2], dim=1)
        up2 = self.invres2(up2)
        # print((dc2.shape, "dc2"))
        # print((up2.shape, "up2"))

        dc3 = self.dconv3(up2)
        up3 = torch.cat([x2, dc3], dim=1)
        up3 = self.invres3(up3)
        # print((dc3.shape, "dc3"))
        # print((up3.shape, "up3"))

        dc4 = self.dconv4(up3)
        up4 = torch.cat([x1, dc4], dim=1)
        up4 = self.invres4(up4)
        # print((dc4.shape, "dc4"))
        # print((up4.shape, "up4"))

        up5 = self.dconv5(up4)

        # if self.invres5 is not None:
        #     up5 = self.invres5(up5)
        # print((up5.shape, "up5"))

        x = up5
        # x = self.softmax(up5)
        # print((x.shape, "softmax"))
        # x = self.sigmoid(x)

        # if self.mode == "eval":
        #     # print("EVALTRUE")
        #     mask_fg = x[0, 0]
        #     mask_bg = x[0, 1]

        #     mask = (1 * mask_bg + (1 - mask_fg)) / 2
        #     # mask = mask_fg
        #     mask[mask < 0.5] = 0
        #     mask[mask >= 0.5] = 1
        #     # mask = torch.logical_not(mask)
        #     return torch.unsqueeze(mask, 0)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def predict(self, img, score_thresh=0.5, tta=False, debug=False, label=None):
        # pad equally left or right, and top or bottom
        img = utils.resize_max_axis(img, config.unet.test_input_size)
        # orig = img.copy()
        img, y_slice, x_slice = utils.pad_equal(img, config.unet.test_input_size)
        # WORKS!
        # assert np.all(orig == img[y_slice, x_slice])

        if label is not None:
            label = utils.resize_max_axis(label, config.unet.test_input_size)
            label, _, _ = utils.pad_equal(label, config.unet.test_input_size)

        is_gray = lambda img: len(img.shape) == 2
        if is_gray(img):
            img = np.repeat(img[..., np.newaxis], 3, -1)
        else:
            # TODO
            pass

        def predict_raw(inputs):
            # normalize
            inputs = inputs.astype(np.float32) / 255.0
            # transpose spatial and color
            inputs = inputs.transpose((2, 0, 1))
            # add batch
            inputs = np.expand_dims(inputs, axis=0)

            inputs = torch.tensor(inputs)
            with torch.no_grad():
                inputs = inputs.to("cuda")
                pred = self.forward(inputs)[0]
                pred = nn.Softmax(dim=0)(pred)
                pred = pred.cpu().numpy()

                bg_pred = pred[0]
                fg_pred = pred[1]

                pred = (fg_pred + 1 - bg_pred) / 2

            return pred

        def threshold_and_multiply(pred):
            pred[pred >= score_thresh] = 1
            pred[pred < score_thresh] = 0

            pred = np.uint8(pred) * 255
            return pred

        if not tta:
            pred = predict_raw(img)
            pred = threshold_and_multiply(pred)
        else:
            orig = img.copy()
            augmented_imgs = [img.copy()]
            for rot in (90, 180, 270):
                img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
                augmented_imgs.append(img.copy())

            img = cv.flip(orig.copy(), +1)
            augmented_imgs.append(img.copy())
            for rot in (90, 180, 270):
                img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
                augmented_imgs.append(img.copy())

            preds = [predict_raw(img) for img in augmented_imgs]
            # de-augment
            preds = np.array(
                [
                    preds[0],
                    cv.rotate(preds[1], cv.ROTATE_90_COUNTERCLOCKWISE),
                    cv.rotate(preds[2], cv.ROTATE_180),
                    cv.rotate(preds[3], cv.ROTATE_90_CLOCKWISE),
                    cv.flip(preds[4], +1),
                    cv.flip(cv.rotate(preds[5], cv.ROTATE_90_COUNTERCLOCKWISE), +1),
                    cv.flip(cv.rotate(preds[6], cv.ROTATE_180), +1),
                    cv.flip(cv.rotate(preds[7], cv.ROTATE_90_CLOCKWISE), +1),
                ]
            )
            pred = preds.mean(axis=0)
            pred = threshold_and_multiply(pred)

        # if debug:
        #     utils.show(pred, pred[y_slice, x_slice])

        if label is not None:
            return pred[y_slice, x_slice], label[y_slice, x_slice] * 255
        else:
            return pred[y_slice, x_slice]

    def unload(self):
        self.cpu()
        # from numba import cuda

        # cuda.select_device(0)
        # cuda.close()


if __name__ == "__main__":
    # Debug
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = MobileNetV2_unet(pre_trained=None)
    net(torch.randn(1, 3, 224, 224))
