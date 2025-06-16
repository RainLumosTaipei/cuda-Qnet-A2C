import torch
import torch.nn.functional as F
import numpy as np

from utils.misc import timeit
from .Qnet import Qnet
from .ReplayBuffer import ReplayBuffer
from tqdm import tqdm
import csv

class DQN:
    
    __lr = 2e-3           # 学习率,控制参数更新步长，过大导致不稳定，过小导致收敛慢
    __num_episodes = 100  # 100次迭代
    __hidden_dim = 128    # DQN 隐藏层神经元数量
    __gamma = 0.98        # 折扣因子：取值范围 [0, 1)，决定未来奖励的折现程度
    __epsilon = 0.01      # 贪婪策略中的探索率, 表示 1% 的概率随机选择动作，99% 的概率选择最优动作
    __target_update = 10  # 目标网络更新频率, 表示每 10 步更新一次目标网络
    __buffer_size = 10000 # 经验池最大容量
    __minimal_size = 500  # 最小经验池大小
    __batch_size = 64     # 每次采样大小,这个数据是合适的

    def __init__(self,state_dim,action_dim,device,env):
        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer(self.__buffer_size)
        # 主 Q 网络，用于选择动作和训练
        self.q_net = Qnet(state_dim, self.__hidden_dim, self.action_dim).to(device)  
        # 目标 Q 网络，提供稳定的目标值
        self.target_q_net = Qnet(state_dim, self.__hidden_dim, self.action_dim).to(device)
        # 使用 Adam 优化器更新主 Q 网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.__lr)
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.env = env

    # 选取动作
    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.__epsilon:
            # 随机选择动作
            action = np.random.randint(self.action_dim)
        else:
            # 将环境返回的状态转换为 PyTorch 张量
            state = torch.as_tensor(
                state, dtype=torch.float, device=self.device
            ).unsqueeze(0) # 添加维度
            # 使用主 Q 网络评估，返回 Q 值最大的动作索引
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        # 提取数据
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        # view(-1, 1)用于转换向量类型
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.__gamma * max_next_q_values * (1 - dones)  # TD误差目标
        
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        # 每执行 target_update 次训练，将主网络的参数复制到目标网络。
        if self.count % self.__target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    @timeit
    def train(self):
        # 记录每一次的迭代结果
        return_list = []
        # 将迭代均分为10份，总迭代次数不变
        num_iterations = 10
        # 100 / 10 = 10
        episodes_per_iter = self.__num_episodes // num_iterations  

        # 以下对每一次迭代进行训练
        for i in range(num_iterations):
            # 使用 tqdm 库显示每个迭代阶段的进度条
            with tqdm(total=episodes_per_iter, desc=f"Iteration {i}") as pbar:
                for i_episode in range(episodes_per_iter):
                    # 每一次迭代都要环境重置，初始化状态和累计回报
                    episode_return = 0
                    state = self.env.reset()[0]
                    done = False

                    while not done:
                        # 以下开始进行对数据的决策，并记录到经验缓冲区
                        # 智能体根据当前状态选择动作
                        action = self.take_action(state)
                        # 执行动作并获取环境反馈
                        next_state, reward, done, _, _ = self.env.step(action)
                        # 将状态转移存入经验回放缓冲区
                        self.replay_buffer.add(state, action, reward, next_state, done)
                        # 更新状态和累计回报
                        state = next_state
                        episode_return += reward


                        # 当经验缓冲区中的样本数量足够时进行采样，并更新指标
                        if self.replay_buffer.size() > self.__minimal_size:
                            # 采样 batch_size 个
                            b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.__batch_size)
                            transition_dict = {
                                "states": b_s,
                                "actions": b_a,
                                "next_states": b_ns,
                                "rewards": b_r,
                                "dones": b_d,
                            }
                            # 使用采样得到的批量样本更新智能体参数
                            self.update(transition_dict)
                    
                    # 完成并记录一次迭代
                    return_list.append(episode_return)
                    
                    # 每 10 次迭代更新一次进度条的统计信息，显示最近的平均回报
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix(
                            {
                                "episode": f"{(episodes_per_iter * i) + i_episode + 1}",
                                # 取列表中最后 10 个元素
                                "return": f"{np.mean(return_list[-10:]):.3f}",
                            }
                        )
                    pbar.update(1)


        with open("result/return.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for val in return_list:
                writer.writerow([val]) 


