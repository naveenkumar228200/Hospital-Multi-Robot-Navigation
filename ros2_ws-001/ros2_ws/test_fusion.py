import torch

from hospital_attention.models.lidar_encoder_v2 import LidarEncoderV2
from hospital_attention.models.depth_encoder import DepthEncoder
from hospital_attention.models.imu_encoder import IMUEncoder
from hospital_attention.models.wheel_encoder import WheelEncoder
from hospital_attention.models.attention_fusion import AttentionFusion

lidar_encoder = LidarEncoderV2()
depth_encoder = DepthEncoder()
imu_encoder = IMUEncoder()
wheel_encoder = WheelEncoder()

fusion = AttentionFusion()

# Dummy sensor inputs
lidar_input = torch.rand(1,3,1024)
depth_input = torch.rand(1,3,224,224)
imu_input = torch.rand(1,10)
wheel_input = torch.rand(1,5)

lidar_feat = lidar_encoder(lidar_input)
depth_feat = depth_encoder(depth_input)
imu_feat = imu_encoder(imu_input)
wheel_feat = wheel_encoder(wheel_input)

fused, weights = fusion(
    lidar_feat,
    depth_feat,
    imu_feat,
    wheel_feat
)

print("LiDAR :", lidar_feat.shape)
print("Depth :", depth_feat.shape)
print("IMU   :", imu_feat.shape)
print("Wheel :", wheel_feat.shape)
print("Fused :", fused.shape)
print("Weights:", weights.shape)
