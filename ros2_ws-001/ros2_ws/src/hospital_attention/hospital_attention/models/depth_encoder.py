import torch
import torch.nn as nn

from torchvision.models import (

    mobilenet_v3_small,

    MobileNet_V3_Small_Weights

)


class DepthEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = mobilenet_v3_small(

            weights=MobileNet_V3_Small_Weights.DEFAULT

        )

        #######################################

        old_conv = backbone.features[0][0]

        new_conv = nn.Conv2d(

            1,

            old_conv.out_channels,

            kernel_size=old_conv.kernel_size,

            stride=old_conv.stride,

            padding=old_conv.padding,

            bias=False

        )

        new_conv.weight.data = (

            old_conv.weight.data.mean(

                dim=1,

                keepdim=True

            )

        )

        backbone.features[0][0] = new_conv

        #######################################

        self.features = backbone.features

        for p in self.features.parameters():

            p.requires_grad = False

        #######################################

        self.pool = nn.AdaptiveAvgPool2d(

            1

        )

        #######################################

        self.projection = nn.Sequential(

            nn.Linear(

                576,

                128

            ),

            nn.ReLU(),

            nn.Linear(

                128,

                64

            )

        )

    ################################################

    def forward(

            self,

            x

    ):

        x = self.features(

            x

        )

        x = self.pool(

            x

        )

        x = x.flatten(

            1

        )

        x = self.projection(

            x

        )

        return x
