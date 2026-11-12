import torch
import torch.nn as nn
from models.monotracker.model.blocks import BasicEncoder
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
img_size = 10
x = torch.randn(1, 4, img_size, img_size).cuda()
y = torch.randn(1, 4, img_size, img_size).cuda()

# model = BasicEncoder(input_dim=3, output_dim=128, norm_fn="instance", dropout=0,
#                                  stride=2, Embed3D=False
#                                  ).cuda()
img = torch.cat([x, y], dim=0)
rgbs_ = img[:, :3]  # torch.Size([2, 3, img_size, img_size])
depths = img[:, 3:].clone()  # torch.Size([2, 1, img_size, img_size])
depth_all = img[:, 3,...]  # torch.Size([2, img_size, img_size])
d_near = depth_all[depth_all>0.01].min().item()
d_far = depth_all[depth_all>0.01].max().item()
depths = (depths - d_near) / (d_far - d_near)
depths_dn = nn.functional.interpolate(
            depths, scale_factor=1.0 / 2, mode="nearest")
print(rgbs_.shape)
print(depths.shape)
print(depth_all.shape)
# output = model(input_)
# print(output.shape)

# to do list
# 表格
# 模型
# 损失
