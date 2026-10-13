import torch

from hospital_attention.models.lidar_encoder_v2 import LidarEncoderV2

model = LidarEncoderV2()

points = torch.rand(
    1,
    3,
    1024
)

with torch.no_grad():

    features = model(points)

print(features.shape)
