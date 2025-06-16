import torch
import torch.nn.functional as F

# 一个简单的两层神经网络（一个隐藏层和一个输出层）
# 输入维度为state_dim，对应环境的状态空间维度
# 隐藏层维度为hidden_dim，使用 ReLU 激活函数
# 输出维度为action_dim，对应环境的动作空间维度
# 输出值表示每个动作的 Q 值（动作价值）
class Qnet(torch.nn.Module):
    """只有一层隐藏层的Q网络"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        # Q 网络的第一个全连接层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)   
        # Q 网络的第二个全连接层
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # 第一层 --> relu --> 第二层
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)