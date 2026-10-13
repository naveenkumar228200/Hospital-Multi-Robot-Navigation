import torch
from torch_geometric.nn import fps

points = torch.rand(1000, 3)
batch = torch.zeros(1000, dtype=torch.long)

idx = fps(points, batch, ratio=0.2)

print(idx.shape)
