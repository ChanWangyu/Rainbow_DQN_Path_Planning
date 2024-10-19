from typing import List, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.



    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


# class Network(nn.Module):
#     def __init__(
#             self,
#             in_dim: int,
#             out_dim: int,
#             atom_size: int,
#             support: torch.Tensor
#     ):
#         """Initialization."""
#         super(Network, self).__init__()
#
#         self.support = support
#         self.out_dim = out_dim
#         self.atom_size = atom_size
#
#         # set common feature layer
#         self.feature_layer = nn.Sequential(
#             nn.Linear(in_dim, 128),
#             nn.ReLU(),
#         )
#
#         # set advantage layer
#         self.advantage_hidden_layer = NoisyLinear(128, 128)
#         self.advantage_layer = NoisyLinear(128, out_dim * atom_size)
#
#         # set value layer
#         self.value_hidden_layer = NoisyLinear(128, 128)
#         self.value_layer = NoisyLinear(128, atom_size)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward method implementation."""
#         dist = self.dist(x)
#         q = torch.sum(dist * self.support, dim=2)
#
#         return q
#
#     def dist(self, x: torch.Tensor) -> torch.Tensor:
#         """Get distribution for atoms."""
#         feature = self.feature_layer(x)
#         adv_hid = F.relu(self.advantage_hidden_layer(feature))
#         val_hid = F.relu(self.value_hidden_layer(feature))
#
#         advantage = self.advantage_layer(adv_hid).view(
#             -1, self.out_dim, self.atom_size
#         )
#         value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
#         q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
#
#         dist = F.softmax(q_atoms, dim=-1)
#         dist = dist.clamp(min=1e-3)  # for avoiding nans
#
#         return dist
#
#     def reset_noise(self):
#         """Reset all noisy layers."""
#         self.advantage_hidden_layer.reset_noise()
#         self.advantage_layer.reset_noise()
#         self.value_hidden_layer.reset_noise()
#         self.value_layer.reset_noise()


class CNNNetwork(nn.Module):
    def __init__(self, grid_size: int, out_dim: int, atom_size: int, support: torch.Tensor):
        """Initialization."""
        super(CNNNetwork, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # 定义卷积层
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 输入为单通道的矩阵
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 个输出通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化
        )

        # 计算展平后的维度
        flattened_size = (grid_size // 2) * (grid_size // 2) * 64

        # 定义全连接层
        self.fc_advantage_hidden = NoisyLinear(flattened_size, 128)
        self.fc_advantage = NoisyLinear(128, out_dim * atom_size)

        self.fc_value_hidden = NoisyLinear(flattened_size, 128)
        self.fc_value = NoisyLinear(128, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        # x 形状为 (batch_size, 1, grid_size, grid_size)
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # 将卷积层输出展平

        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        adv_hid = F.relu(self.fc_advantage_hidden(x))
        val_hid = F.relu(self.fc_value_hidden(x))

        advantage = self.fc_advantage(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.fc_value(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # 避免出现 nan

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.fc_advantage_hidden.reset_noise()
        self.fc_advantage.reset_noise()
        self.fc_value_hidden.reset_noise()
        self.fc_value.reset_noise()


GRID_SIZE = 8  # 网格大小 (8x8)
OUT_DIM = 4  # 动作数量 (输出维度)
ATOM_SIZE = 51  # 分布式Q学习中的原子数量
V_MIN = -10  # 支持向量的最小值
V_MAX = 10  # 支持向量的最大值

# 创建支持向量
support = torch.linspace(V_MIN, V_MAX, ATOM_SIZE)

from Rainbow_DQN_Path_Planning.env import GridmapEnv

def main():
    # 创建 GridmapEnv 实例
    env = GridmapEnv(grid_size=(GRID_SIZE, GRID_SIZE), obstacle_ratio=0.2, seed=34)
    obs, mask = env.reset()  # 重置环境，获取初始观察值和动作掩码

    # 初始化网络
    model = CNNNetwork(grid_size=GRID_SIZE, out_dim=OUT_DIM, atom_size=ATOM_SIZE, support=support)

    # 将环境的观察值转换为适合模型输入的张量
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)  # 添加 batch 维度和 channel 维度

    print(obs_tensor.shape)

    # 使用模型预测 Q 值
    with torch.no_grad():  # 不需要计算梯度
        q_values = model(obs_tensor)  # 通过网络获得 Q 值
        print("Q Values:", q_values)

    # 选择动作（根据 epsilon-greedy 策略）
    action = q_values.argmax(dim=1).item()  # 选择具有最大 Q 值的动作
    print("Selected Action:", action)

    # 执行动作并获取下一个状态
    next_obs, next_mask, reward, done, info = env.step(action)
    print("Next Observation:"'\n', next_obs)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)

    # 在结束时关闭环境
    env.close()

if __name__ == "__main__":
    main()