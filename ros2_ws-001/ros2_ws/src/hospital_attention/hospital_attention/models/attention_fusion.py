import torch
import torch.nn as nn


class AttentionFusion(nn.Module):

    def __init__(self):

        super().__init__()

        ##################################################
        # IMU
        ##################################################

        self.imu_proj = nn.Linear(

            32,

            64

        )

        ##################################################
        # Wheel
        ##################################################

        self.wheel_proj = nn.Linear(

            32,

            64

        )

        ##################################################
        # Fusion
        ##################################################

        self.fusion = nn.Sequential(

            nn.Linear(

                256,

                256

            ),

            nn.ReLU(),

            nn.Linear(

                256,

                128

            )

        )

    def forward(

            self,

            lidar_feat,

            depth_feat,

            imu_feat,

            wheel_feat

    ):

        imu_feat = self.imu_proj(

            imu_feat

        )

        wheel_feat = self.wheel_proj(

            wheel_feat

        )

        x = torch.cat(

            [

                lidar_feat,

                depth_feat,

                imu_feat,

                wheel_feat

            ],

            dim=1

        )

        fused = self.fusion(

            x

        )

        return fused
