import math
import numpy as np

class AdamOptimizer:
    """纯Python实现的Adam优化器"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # 初始化动量存储
        self.m = []
        self.v = []
        for layer in params:
            weights, biases = layer
            self.m.append(
                (
                    [[0.0] * len(weights[0]) for _ in range(len(weights))],
                    [0.0] * len(biases),
                )
            )
            self.v.append(
                (
                    [[0.0] * len(weights[0]) for _ in range(len(weights))],
                    [0.0] * len(biases),
                )
            )

    def step(self, gradients, params):
        """执行参数更新"""
        self.t += 1
        for layer_idx in range(len(params)):
            # 权重更新
            for i in range(len(params[layer_idx][0])):
                for j in range(len(params[layer_idx][0][i])):
                    key = ""
                    if layer_idx == 0:
                        key = "actor.W1"
                    elif layer_idx == 1:
                        key = "actor.W2"
                    elif layer_idx == 2:
                        key = "critic.W1"
                    elif layer_idx == 3:
                        key = "critic.W2"

                    g = gradients[key][i][j]

                    self.m[layer_idx][0][i][j] = (
                        self.beta1 * self.m[layer_idx][0][i][j] + (1 - self.beta1) * g
                    )
                    self.v[layer_idx][0][i][j] = (
                        self.beta2 * self.v[layer_idx][0][i][j]
                        + (1 - self.beta2) * g**2
                    )

                    m_hat = self.m[layer_idx][0][i][j] / (1 - self.beta1**self.t)
                    v_hat = self.v[layer_idx][0][i][j] / (1 - self.beta2**self.t)

                    # 数值稳定性检查
                    if v_hat < 0:
                        v_hat = 1e-8  # 设置为小正数

                    if np.isnan(m_hat) or np.isinf(m_hat):
                        m_hat = 0.0

                    if np.isnan(v_hat) or np.isinf(v_hat):
                        v_hat = 1e-8  # 设置为小正数

                    params[layer_idx][0][i][j] -= (
                        self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
                    )

            # 偏置更新
            for i in range(len(params[layer_idx][1])):
                key = ""
                if layer_idx == 0:
                    key = "actor.b1"
                elif layer_idx == 1:
                    key = "actor.b2"
                elif layer_idx == 2:
                    key = "critic.b1"
                elif layer_idx == 3:
                    key = "critic.b2"

                g = gradients[key][i]

                self.m[layer_idx][1][i] = (
                    self.beta1 * self.m[layer_idx][1][i] + (1 - self.beta1) * g
                )
                self.v[layer_idx][1][i] = (
                    self.beta2 * self.v[layer_idx][1][i] + (1 - self.beta2) * g**2
                )

                m_hat = self.m[layer_idx][1][i] / (1 - self.beta1**self.t)
                v_hat = self.v[layer_idx][1][i] / (1 - self.beta2**self.t)

                params[layer_idx][1][i] -= (
                    self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
                )
