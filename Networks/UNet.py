import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config


class ConvBlock(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)
        self.norm = nn.BatchNorm2d(self.out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = super().forward(input)
        y = self.norm(y)
        return F.relu(y)


class UBlock(nn.Module):
    def __init__(self, in_channels, branch, out_channels) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            ConvBlock(in_channels, out_channels, 3, 1, 1),
            ConvBlock(out_channels, out_channels, 3, 1, 1),
        )
        self.upsample = nn.Sequential(
            ConvBlock(out_channels*2, out_channels*4, 1, 1, 0),
            nn.PixelShuffle(2),
        )
        self.branch = branch
        self.post = nn.Sequential(
            ConvBlock(out_channels*2, out_channels, 3, 1, 1),
            ConvBlock(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.pre(input)
        branch = F.max_pool2d(output, 2, 2)
        branch = self.branch(branch)
        branch = self.upsample(branch)
        output = torch.cat((output, branch), dim=1)
        output = self.post(output)
        return output


class UNet(nn.Sequential):
    def __init__(self, in_channels, num_classes, ref_channels=64) -> None:
        UStructure = \
            UBlock(
                in_channels,
                UBlock(
                    ref_channels,
                    UBlock(
                        2*ref_channels,
                        UBlock(
                            4*ref_channels,
                            nn.Sequential(
                                ConvBlock(
                                    8*ref_channels, 16 * ref_channels, 3, 1, 1
                                ),
                                ConvBlock(
                                    16*ref_channels, 16 * ref_channels, 3, 1, 1
                                ),
                            ),
                            8*ref_channels,
                        ),
                        4*ref_channels,
                    ),
                    2*ref_channels,
                ),
                ref_channels,
            )
        mapping = nn.Conv2d(ref_channels, num_classes, 1, 1, 0)
        softmax = nn.Softmax(dim=1)
        super().__init__(UStructure, mapping, softmax)


if __name__ == "__main__":
    unet = UNet(3, 22).to("cuda")
    print(str(unet))
    a = torch.zeros((16, 3, 384, 384)).to("cuda")
    while True:
        print(a.shape, unet.forward(a).shape)
    pass
