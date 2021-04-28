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

# logging.basicConfig(level=logging.DEBUG)

# paper implementation
class MobileNetV2_unet(nn.Module):
    def __init__(
        self,
        n_classes=1000,
        input_size=224,
        channels=3,
        pretrained="weights/mobilenet_v2.pth.tar",
        mode="train",
    ):
        super(MobileNetV2_unet, self).__init__()

        self.mode = mode
        self.input_size = input_size

        self.backbone = MobileNetV2(
            n_classes=1000, input_size=input_size, channels=channels, width_mult=1.0
        )

        # input_channel, last_channel
        # self.dense0 = conv_1x1_bn(320, 1280)

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        # TODO upsample bilinear
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        # dimfred
        self.dconv5 = nn.ConvTranspose2d(16, n_classes, 4, padding=1, stride=2)
        # self.invres5 = InvertedResidual(16, 8, 1, 6)
        # self.conv_last = nn.Conv2d(8, 3, 1)

        self.softmax = nn.Softmax(dim=1)

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
        print((x.shape, "input"))
        for n in range(0, 2):
            x = self.backbone.features[n](x)

        x1 = x
        print((x1.shape, "x1"))

        for n in range(2, 4):
            x = self.backbone.features[n](x)
        x2 = x
        print((x2.shape, "x2"))

        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x
        print((x3.shape, "x3"))

        for n in range(7, 14):
            x = self.backbone.features[n](x)
        x4 = x
        print((x4.shape, "x4"))

        # TODO 1x1 layer removed hence 18 instead of 19
        for n in range(14, 18):
            x = self.backbone.features[n](x)

        x = self.backbone.conv(x)
        x5 = x
        print((x5.shape, "x5"))

        dc1 = self.dconv1(x)
        up1 = torch.cat([x4, dc1], dim=1)
        up1 = self.invres1(up1)
        print((dc1.shape, "dc1"))
        print((up1.shape, "up1"))

        dc2 = self.dconv2(up1)
        up2 = torch.cat([x3, dc2], dim=1)
        up2 = self.invres2(up2)
        print((dc2.shape, "dc2"))
        print((up2.shape, "up2"))

        dc3 = self.dconv3(up2)
        up3 = torch.cat([x2, dc3], dim=1)
        up3 = self.invres3(up3)
        print((dc3.shape, "dc3"))
        print((up3.shape, "up3"))

        dc4 = self.dconv4(up3)
        up4 = torch.cat([x1, dc4], dim=1)
        up4 = self.invres4(up4)
        print((dc4.shape, "dc4"))
        print((up4.shape, "up4"))

        up5 = self.dconv5(up4)
        print((up5.shape, "up5"))

        x = self.softmax(up5)
        print((x.shape, "softmax"))

        if self.mode == "eval":
            print("EVALTRUE")
            mask_fg = x[0, 0]
            mask_bg = x[0, 1]

            mask = (1 * mask_bg + (1 - mask_fg)) / 2
            # mask = mask_fg
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            # mask = torch.logical_not(mask)
            return torch.unsqueeze(mask, 0)

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

    def predict(self, img):
        img = cv.resize(img, (self.input_size, self.input_size))

        is_gray = lambda img: len(img.shape) == 2
        if is_gray(img):
            img = np.repeat(img[..., np.newaxis], 3, -1)
        else:
            # TODO
            pass

        inputs = img
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
            pred = np.uint8(pred.cpu().numpy() * 255)

        return pred

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
