import torch
import random
import numpy as np

from utils.misc import timeit
from .Qnet import Qnet
from .ReplayBuffer import ReplayBuffer
from .AdamOptimizer import AdamOptimizer
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

    def __init__(self,state_dim, action_dim, device, env):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, self.__hidden_dim, self.action_dim)
        self.target_q_net = Qnet(state_dim, self.__hidden_dim, self.action_dim)
        self.optimizer = AdamOptimizer(self.q_net.parameters(), lr=self.__lr)
        self.count = 0
        self.replay_buffer = ReplayBuffer(self.__buffer_size)
        self.device = device  
        self.env = env

    def take_action(self, state):
        if random.random() < self.__epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            q_values = self.q_net(list(state))
            return q_values.index(max(q_values))

    def update(self, transition_dict):
        # 转换数据格式
        states = [list(s) for s in transition_dict["states"]] # n x a
        actions = transition_dict["actions"]
        rewards = transition_dict["rewards"]  # 1 x n
        next_states = [list(ns) for ns in transition_dict["next_states"]]
        dones = transition_dict["dones"]  # 1 x n

        batch_size = len(states)

        # 计算当前Q值
        # 网络对当前状态的所有动作的 Q 值预测
        q_values = self.q_net(states) # n x c
        # 从每个样本的 Q 值中提取出实际执行动作对应的 Q 值
        q_selected = [q_values[i][actions[i]] for i in range(batch_size)] # 1 x n

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states) # n x c
        # 选择每个下一状态的最大 Q 值 (贪婪策略)
        max_next_q = [max(q) for q in next_q_values] # 1 x n
        # 根据贝尔曼方程计算目标 Q 值
        q_targets = [
            r + self.__gamma * m * (1 - d)   for r, m, d in zip(rewards, max_next_q, dones)
        ]   # 1 x n

        # 计算梯度, 这里使用均方误差
        # Loss = 1/N × Σ(Q预测 - Q目标)²
        # ∂Loss/∂Q预测 = 2/N × (Q预测 - Q目标)
        gradients = [] # n x c
        for i in range(batch_size):
            grad = [0.0] * self.action_dim  # 1 x c
            grad[actions[i]] = 2 * (q_selected[i] - q_targets[i]) / batch_size # 计算梯度
            gradients.append(grad)

        # 反向传播
        # 重置原有梯度
        self.q_net.zero_grad()
        # 计算反向传播
        self.q_net.backward(states, gradients)

        # 参数更新
        self.optimizer.step(self.q_net._gradients, self.q_net.parameters())

        # 目标网络更新
        if self.count % self.__target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
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


