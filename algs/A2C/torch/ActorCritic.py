import torch.nn as nn
from torch.distributions import Categorical


# 演员评论家模型
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        # 评价网络
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),  # 第一层
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # 第二层，输出预估价值
        )

        # 动作网络
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),  # 第一层
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),  # 第二层
            nn.Softmax(dim=0),  # 将输出转换为概率分布
        )

    # 传播
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        # 对动作概率进行封装, 包括以下几个函数
        # dist.sample() 返回最大概率的动作
        # dist.entropy() 计算熵
        # dist.log_prob(action) 计算动作的对数概率
        dist = Categorical(probs)
        return dist, value  # 预估值，动作概率