import ctypes
import math
import random
from collections import defaultdict
import copy

lib = ctypes.CDLL("./lib/qnet.dll")

# 定义CUDA函数参数类型
lib.cuda_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]

lib.cuda_backward.argtypes = [
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
    """将二维列表转换为一维float数组"""
    flat = (
        [item    for sublist in arr    for item in sublist]  if isinstance(arr[0], list)
        else arr
    )
    # 创建一个长度为n的 C 风格浮点型数组类型
    # 使用*操作符将 Python 列表flat解包为位置参数。
    return (ctypes.c_float * len(flat))( *flat )


def flatten_weights(w):
    return [item   for sublist in w   for item in sublist]

# 转置后展平矩阵
def flatten_weights_after_tranpose(w):
    # 先转置矩阵，再展平为一维数组
    transposed = list(map(list, zip(*w)))  # 转置矩阵
    return [item   for row in transposed  for item in row]  # 按行展平

class Qnet:
    def __init__(self, state_dim, hidden_dim, action_dim):
        # 参数初始化（保持原有接口）
        self.state_dim = state_dim  # a
        self.hidden_dim = hidden_dim # b
        self.action_dim = action_dim # c

        # 权重参数（二维列表形式存储）
        # Xavier 初始化，旨在保持信号在神经网络各层间的方差一致性，避免梯度消失或爆炸。
        # 权重矩阵中的每个元素将从均匀分布中随机采样。
        # y = x * W^T + b  注意权重矩阵需要转置，因此与矩阵乘法不同
        # b x a
        self.fc1_weight = [
            [
                random.uniform(
                    -math.sqrt(6 / (state_dim + hidden_dim)),
                    math.sqrt(6 / (state_dim + hidden_dim)),
                )
                for _ in range(state_dim)
            ]
            for _ in range(hidden_dim)
        ]
        # 创建一个长度为 hidden_dim 的列表，每个元素初始化为 0.0
        # 1 x b
        self.fc1_bias = [0.0] * hidden_dim


        # c x b
        self.fc2_weight = [
            [
                random.uniform(
                    -math.sqrt(6 / (hidden_dim + action_dim)),
                    math.sqrt(6 / (hidden_dim + action_dim)),
                )
                for _ in range(hidden_dim)
            ]
            for _ in range(action_dim)
        ]
        # 1 x c
        self.fc2_bias = [0.0] * action_dim

        # 兼容接口，梯度矩阵  n x c
        self._gradients = defaultdict(list)

        # 参数矩阵列表
        self._parameters = [
            (self.fc1_weight, self.fc1_bias),
            (self.fc2_weight, self.fc2_bias),
        ]

    def forward(self, x):
        # 转换为CUDA需要的格式
        # 输入为 n x a
        # 获取输入个数 n
        batch_size = 1 if not isinstance(x[0], list) else len(x)
        input_array = list_to_float_ptr(x)

        # 准备CUDA输入输出
        # n x c
        output_array = (ctypes.c_float * (batch_size * self.action_dim))()

        # 转换权重参数
        fc1_w_flat = flatten_weights_after_tranpose(self.fc1_weight)
        fc1_b_flat = (ctypes.c_float * len(self.fc1_bias))(*self.fc1_bias)

        fc2_w_flat = flatten_weights_after_tranpose(self.fc2_weight)
        fc2_b_flat = (ctypes.c_float * len(self.fc2_bias))(*self.fc2_bias)

        # 调用CUDA前向传播
        lib.cuda_forward(
            input_array,
            batch_size,       # n
            self.state_dim,  # a
            self.hidden_dim, # b
            self.action_dim, # c
            (ctypes.c_float * len(fc1_w_flat))(*fc1_w_flat), 
            fc1_b_flat,
            (ctypes.c_float * len(fc2_w_flat))(*fc2_w_flat),
            fc2_b_flat,
            output_array,  # 输出
        )

        # 转换输出格式
        # n x c
        output = [
            [output_array[i * self.action_dim + j] for j in range(self.action_dim)]  # c
            for i in range(batch_size)  # n
        ]
        return output[0] if batch_size == 1 else output

    def backward(self, x, grad_output):
        # 转换输入格式
        x = x if isinstance(x[0], list) else [x] # n x a
        batch_size = len(x) # n

        # 前向传播获取hidden层的值，这里应该提前保存，不应该重复计算
        # 这里简化处理，实际应分离前向传播步骤
        hidden = []  # n x b
        for sample in x:  # n x a --> 1 x a
            h = []
            for i in range(self.hidden_dim): # b
                weighted_sum = sum(
                    x_j * w_ij for x_j, w_ij in zip(sample, self.fc1_weight[i]) #  b x a --> 1 x a
                )
                # ReLU
                h.append(max(0, weighted_sum + self.fc1_bias[i])) # 得到 1 x b 的一个分量
            # h: 1 x b
            hidden.append(h)

        # 准备CUDA输入
        input_array = list_to_float_ptr(x)  # n x a
        hidden_array = list_to_float_ptr(hidden) # n x b
        grad_output_array = list_to_float_ptr(grad_output) # n x c
        fc2_w_flat = flatten_weights_after_tranpose(self.fc2_weight) # b x c

        # 准备梯度存储
        grad_fc1_w = [0.0] * (self.hidden_dim * self.state_dim) # b x a
        grad_fc1_b = [0.0] * self.hidden_dim # 1 x b
        grad_fc2_w = [0.0] * (self.action_dim * self.hidden_dim)  # c x b
        grad_fc2_b = [0.0] * self.action_dim # 1 x c

        # 调用CUDA反向传播
        lib.cuda_backward(
            input_array,
            hidden_array,
            grad_output_array,
            (ctypes.c_float * len(fc2_w_flat))(*fc2_w_flat),
            self.state_dim,
            self.hidden_dim,
            self.action_dim,
            batch_size,
            (ctypes.c_float * len(grad_fc1_w))(*grad_fc1_w),
            (ctypes.c_float * len(grad_fc1_b))(*grad_fc1_b),
            (ctypes.c_float * len(grad_fc2_w))(*grad_fc2_w),
            (ctypes.c_float * len(grad_fc2_b))(*grad_fc2_b),
        )

        # 转换梯度格式
        # 这里需要先转置矩阵 TODO
        self._gradients["fc1.weight"] = [  # b x a
            grad_fc1_w[i * self.state_dim : (i + 1) * self.state_dim]
            for i in range(self.hidden_dim)
        ]
        self._gradients["fc1.bias"] = grad_fc1_b # 1 x b
        self._gradients["fc2.weight"] = [  # c x b
            grad_fc2_w[i * self.hidden_dim : (i + 1) * self.hidden_dim]
            for i in range(self.action_dim)
        ]
        self._gradients["fc2.bias"] = grad_fc2_b  # 1 x c

    def parameters(self):
        """返回所有参数（兼容PyTorch接口）"""
        return [(self.fc1_weight, self.fc1_bias), (self.fc2_weight, self.fc2_bias)]

    def load_state_dict(self, state_dict):
        """加载参数（深拷贝）"""
        self.fc1_weight = copy.deepcopy(state_dict["fc1.weight"])
        self.fc1_bias = copy.deepcopy(state_dict["fc1.bias"])
        self.fc2_weight = copy.deepcopy(state_dict["fc2.weight"])
        self.fc2_bias = copy.deepcopy(state_dict["fc2.bias"])

    def state_dict(self):
        """返回参数字典（兼容PyTorch接口）"""
        return {
            "fc1.weight": copy.deepcopy(self.fc1_weight),
            "fc1.bias": copy.deepcopy(self.fc1_bias),
            "fc2.weight": copy.deepcopy(self.fc2_weight),
            "fc2.bias": copy.deepcopy(self.fc2_bias),
        }

    def zero_grad(self):
        """梯度清零"""
        self._gradients = defaultdict(list)

    def to(self, device):
        """兼容设备转移方法"""
        return self

    def __call__(self, x):
        """模拟PyTorch的调用方式"""
        return self.forward(x)
