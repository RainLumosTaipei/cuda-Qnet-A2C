import random, math, ctypes
import numpy as np
from collections import defaultdict

import torch
from torch.distributions import Categorical

lib = ctypes.CDLL("./lib/a2c.dll")

lib.actor_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)
]

lib.critic_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)
]

lib.actor_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]

lib.critic_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]

def list_to_float_ptr(arr):
    flat = (
        [item    for sublist in arr    for item in sublist]  if isinstance(arr[0], list)
        else arr
    )
    return (ctypes.c_float * len(flat))( *flat )


def init_weights(rows, cols):
    return np.random.randn(rows, cols).astype(np.float32)

class Network:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.output_dim = output_size

        # 使用NumPy数组存储权重和偏置
        self.W1 = init_weights(self.hidden_dim, self.input_dim)
        self.W2 = init_weights(self.output_dim, self.hidden_dim)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.b2 = np.zeros(self.output_dim, dtype=np.float32)
        self.hidden = np.zeros(self.hidden_dim, dtype=np.float32)

        # 梯度也使用NumPy数组
        self.dW1 = np.zeros_like(self.W1)
        self.dW2 = np.zeros_like(self.W2)
        self.db1 = np.zeros_like(self.b1)
        self.db2 = np.zeros_like(self.b2)

    def zero_grad(self):
        self.dW1.fill(0.0)
        self.dW2.fill(0.0)
        self.db1.fill(0.0)
        self.db2.fill(0.0)

    def hidden_ptr(self):
        return self.hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def grad_ptr(self):
        return [
            self.dW1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.dW2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.db1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.db2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ]

    def para_ptr(self):
        return [
            self.W1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.W2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.b1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.b2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ]

    def dim(self):
        return [self.input_dim, self.hidden_dim, self.output_dim]


# 演员评论家模型
class ActorCritic():

    # input: A , hidden : B , output: C
    def __init__(self, num_input, num_output, num_hidden):
        self.critic = Network(num_input, num_hidden, 1)
        self.actor = Network(num_input, num_hidden, num_output)
        self._gradients = defaultdict(list)
        self._gradients["actor.W1"] = self.actor.dW1
        self._gradients["actor.b1"] = self.actor.db1
        self._gradients["actor.W2"] = self.actor.dW2
        self._gradients["actor.b2"] = self.actor.db2
        self._gradients["critic.W1"] = self.critic.dW1
        self._gradients["critic.b1"] = self.critic.db1
        self._gradients["critic.W2"] = self.critic.dW2
        self._gradients["critic.b2"] = self.critic.db2
        self._parameters = [(self.actor.W1, self.actor.b1),
                            (self.actor.W2, self.actor.b2),
                            (self.critic.W1, self.critic.b1),
                            (self.critic.W2, self.critic.b2)]

        # 评价网络
        # self.critic = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size), # 第一层
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 1)   # 第二层，输出预估价值
        # )

        # 动作网络
        # self.actor = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size), # 第一层
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, num_outputs),    # 第二层
        #     nn.Softmax(dim=0),  # 将输出转换为概率分布
        # )

    def actor_forward(self, batch_size, input_ptr):
        # n x c
        output_ptr = (ctypes.c_float * (batch_size * self.actor.output_dim))()
        self.actor.hidden = np.zeros(batch_size * self.actor.hidden_dim, dtype=np.float32)

        lib.actor_forward(
            input_ptr,
            self.actor.hidden_ptr(),
            output_ptr,
            batch_size,      # n
            *self.actor.dim(),
            *self.actor.para_ptr()
        )

        # n x c
        output = [
            [output_ptr[i * self.actor.output_dim + j] for j in range(self.actor.output_dim)]  # c
            for i in range(batch_size)  # n
        ]
        return output[0] if batch_size == 1 else output

    def critic_forward(self, batch_size, input_ptr):
        # n x c
        output_ptr = (ctypes.c_float * (batch_size * self.critic.output_dim))()
        self.critic.hidden = np.zeros(batch_size * self.critic.hidden_dim, dtype=np.float32)

        lib.critic_forward(
            input_ptr,
            self.critic.hidden_ptr(),
            output_ptr,
            batch_size,  # n
            *self.critic.dim(),
            *self.critic.para_ptr()
        )

        # n x c
        output = [
            [output_ptr[i * self.critic.output_dim + j] for j in range(self.critic.output_dim)]  # c
            for i in range(batch_size)  # n
        ]
        return output[0] if batch_size == 1 else output

    # 传播
    def forward(self, x):
        batch_size = 1 if not isinstance(x[0], list) else len(x)
        input_ptr = list_to_float_ptr(x)

        value = self.critic_forward(batch_size, input_ptr)
        probs = self.actor_forward(batch_size, input_ptr)

        dist = Categorical(torch.tensor(probs))
        return dist, value  # 预估值，动作概率

    def actor_backward(self, batch_size, input_ptr, output_ptr, grad_output_ptr):

        lib.actor_backward(
            input_ptr,
            self.actor.hidden_ptr(),
            output_ptr,
            grad_output_ptr,
            self.actor.W2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size,
            *self.actor.dim(),
            *self.actor.grad_ptr()
        )

    def critic_backward(self, batch_size, input_ptr, grad_output_ptr):

        lib.critic_backward(
            input_ptr,
            self.critic.hidden_ptr(),
            grad_output_ptr,
            self.critic.W2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size,
            *self.critic.dim(),
            *self.critic.grad_ptr()
        )


    def backward(self, x, y, actor_loss, critic_loss):
        x = x if isinstance(x[0], list) else [x]  # n x a
        batch_size = len(x)  # n
        input_ptr = list_to_float_ptr(x)
        output_ptr = list_to_float_ptr(y)
        actor_loss = actor_loss[:len(y)]
        actor_loss_ptr = list_to_float_ptr(actor_loss)

        self.actor_backward(batch_size, input_ptr, output_ptr, actor_loss_ptr)
        critic_loss_ptr = ctypes.c_float(critic_loss)
        self.critic_backward(batch_size, input_ptr, critic_loss_ptr)

        return

    def gradients(self):
        return self._gradients

    def parameters(self):
        return self._parameters

    def zero_grad(self):
        self.critic.zero_grad()
        self.actor.zero_grad()

    def to(self, device):
        return self

    def __call__(self, x):
        return self.forward(x)