
import collections
import random

import numpy as np

# 存放历史经验的缓冲区
class ReplayBuffer:
    """经验回放池"""

    # 有最大容量
    # 当队列中的元素数量超过 capacity 时，最旧的元素会被自动删除（从队列左侧弹出），以保持队列的最大长度。
    # 这是一种常见的先进先出 (FIFO) 缓存策略，在强化学习的经验回放缓冲区中经常使用。
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    # 将数据加入buffer
    def add(self, state, action, reward, next_state, done):  
        self.buffer.append((state, action, reward, next_state, done))

    # 每次更新智能体参数时，不是使用单个样本，
    # 而是从缓冲区中随机抽取 batch_size 个样本组成一个小批量
    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # 目前buffer中数据的数量
    def size(self):  
        return len(self.buffer)
