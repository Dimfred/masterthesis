import torchvision as tv
import torch.nn as nn


class AdaptedMobileNetV3(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AdaptedMobileNetV3, self).__init__()

        # self.conv = nn.Conv2d(1, 3, 3, padding=1)
        self.model = tv.models.segmentation.deeplabv3_mobilenet_v3_large(*args, **kwargs)

    def forward(self, x):
        # x = self.conv(x)
        return self.model.forward(x)["out"]
