import sys
import torch
import torch.nn as nn

sys.path.append(
    "/home/lucifer/ros2_ws/src/hospital_attention/hospital_attention/third_party/Pointnet_Pointnet2_pytorch/models"
)

from pointnet2_cls_ssg import get_model


class LidarEncoderV2(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = get_model(

            num_class=40,

            normal_channel=False

        )

        self.feature_head = nn.Sequential(

            nn.Linear(1024, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64)

        )

    def forward(self, xyz):

        ################################################
        # xyz arrives as (B,N,3)
        ################################################

        xyz = xyz.permute(

            0,

            2,

            1

        )

        ################################################
        # now xyz = (B,3,N)
        ################################################

        B = xyz.shape[0]

        l1_xyz, l1_points = self.backbone.sa1(

            xyz,

            None

        )

        l2_xyz, l2_points = self.backbone.sa2(

            l1_xyz,

            l1_points

        )

        l3_xyz, l3_points = self.backbone.sa3(

            l2_xyz,

            l2_points

        )

        ################################################

        x = l3_points.reshape(

            B,

            1024

        )

        ################################################

        x = self.feature_head(

            x

        )

        return x
