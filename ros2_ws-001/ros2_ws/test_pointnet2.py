import torch
import sys

sys.path.append(
    "/home/lucifer/ros2_ws/src/hospital_attention/hospital_attention/third_party/Pointnet_Pointnet2_pytorch/models"
)

from pointnet2_cls_ssg import get_model

model = get_model(
    num_class=40,
    normal_channel=False
)

model.eval()

points = torch.rand(
    1,
    3,
    1024
)

with torch.no_grad():
    out, features = model(points)

print("Classification output:", out.shape)
print("Feature output:", features.shape)
