import math

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
                    g = gradients["fc1.weight" if layer_idx == 0 else "fc2.weight"][i][
                        j
                    ]

                    self.m[layer_idx][0][i][j] = (
                        self.beta1 * self.m[layer_idx][0][i][j] + (1 - self.beta1) * g
                    )
                    self.v[layer_idx][0][i][j] = (
                        self.beta2 * self.v[layer_idx][0][i][j]
                        + (1 - self.beta2) * g**2
                    )

                    m_hat = self.m[layer_idx][0][i][j] / (1 - self.beta1**self.t)
                    v_hat = self.v[layer_idx][0][i][j] / (1 - self.beta2**self.t)

                    params[layer_idx][0][i][j] -= (
                        self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
                    )

            # 偏置更新
            for i in range(len(params[layer_idx][1])):
                g = gradients["fc1.bias" if layer_idx == 0 else "fc2.bias"][i]

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
