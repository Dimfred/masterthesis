import logging
import math
import sys

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from nets.MobileNetV3 import MobileNetV3, InvertedResidual
from nets.MobileNetV3 import conv_1x1_bn

logging.basicConfig(level=logging.DEBUG)

# paper implementation
class MobileNetV3_unet(nn.Module):
    def __init__(
        self,
        n_classes=1000,
        input_size=224,
        channels=3,
        mode="train",
    ):
        super(MobileNetV3_unet, self).__init__()

        self.mode = mode
        self.backbone = MobileNetV3(
            n_classes=1000, input_size=input_size, channels=channels, width_mult=1.0
        )

        self.dconv1 = nn.ConvTranspose2d(960, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.dconv5 = nn.ConvTranspose2d(16, n_classes, 4, padding=1, stride=2)
        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

    def forward(self, x, *args, **kwargs):
        logging.debug((x.shape, "input"))
        for n in range(0, len(self.backbone.features)):
            x = self.backbone.features[n](x)
            logging.debug((n, x.shape, "n"))

        x = self.backbone.conv(x)
        logging.debug((x.shape, "conv"))

        return x

        # logging.debug((x.shape, "input"))
        # for n in range(0, 2):
        #     x = self.backbone.features[n](x)

        # x1 = x
        # logging.debug((x1.shape, "x1"))

        # for n in range(2, 4):
        #     x = self.backbone.features[n](x)
        # x2 = x
        # logging.debug((x2.shape, "x2"))

        # for n in range(4, 7):
        #     x = self.backbone.features[n](x)
        # x3 = x
        # logging.debug((x3.shape, "x3"))

        # for n in range(7, 14):
        #     x = self.backbone.features[n](x)
        # x4 = x
        # logging.debug((x4.shape, "x4"))

        # # TODO 1x1 layer removed hence 18 instead of 19
        # for n in range(14, 18):
        #     x = self.backbone.features[n](x)

        # x = self.backbone.conv(x)
        # x5 = x
        # logging.debug((x5.shape, "x5"))

        # dc1 = self.dconv1(x)
        # up1 = torch.cat([x4, dc1], dim=1)
        # up1 = self.invres1(up1)
        # logging.debug((dc1.shape, "dc1"))
        # logging.debug((up1.shape, "up1"))

        # dc2 = self.dconv2(up1)
        # up2 = torch.cat([x3, dc2], dim=1)
        # up2 = self.invres2(up2)
        # logging.debug((dc2.shape, "dc2"))
        # logging.debug((up2.shape, "up2"))

        # dc3 = self.dconv3(up2)
        # up3 = torch.cat([x2, dc3], dim=1)
        # up3 = self.invres3(up3)
        # logging.debug((dc3.shape, "dc3"))
        # logging.debug((up3.shape, "up3"))

        # dc4 = self.dconv4(up3)
        # up4 = torch.cat([x1, dc4], dim=1)
        # up4 = self.invres4(up4)
        # logging.debug((dc4.shape, "dc4"))
        # logging.debug((up4.shape, "up4"))

        # up5 = self.dconv5(up4)
        # logging.debug((up5.shape, "up5"))

        # x = self.softmax(up5)
        # logging.debug((x.shape, "softmax"))

        # if self.mode == "eval":
        #     mask_bg = x[0, 0]
        #     mask_fg = x[0, 1]

        #     mask = (1 * mask_bg + (1 - mask_fg)) / 2
        #     # mask = mask_fg
        #     mask[mask < 0.5] = 0
        #     mask[mask >= 0.5] = 1
        #     mask = torch.logical_not(mask)
        #     return torch.unsqueeze(mask, 0)

        # return x

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


if __name__ == "__main__":
    # Debug
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = MobileNetV3_unet(n_classes=2)
    net(torch.randn(1, 3, 256, 256))
