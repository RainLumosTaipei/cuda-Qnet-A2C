# 基于CUDA加速的深度强化学习

[![CUDA](https://img.shields.io/badge/CUDA-11.3-green.svg)](https://developer.nvidia.com/cuda-11.3.0-download-archive)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10.0-orange.svg)](https://pytorch.org/)
[![Gym](https://img.shields.io/badge/Gym-latest-blue.svg)](https://www.gymlibrary.dev/)

## 项目简介

一个手动编写CUDA的深度Q网络（DQN）实现，旨在通过CUDA加速深度强化学习算法的训练过程。

## 性能对比

在CartPole-v1训练任务上，与纯CPU实现相比：

- 端到端训练时间：10倍加速（CPU: 27.5分钟 vs. CUDA: 2.75分钟）
- 长回合训练（第7次迭代）：17倍加速（CPU: 581秒 vs. CUDA: 34秒）

![实验结果](https://img.oss.logan.ren/2025/04/28/20-11-40.jpg)

## 环境要求

- CUDA 11.3+
- Python 3.8+
- PyTorch 1.10.0
- NVIDIA GPU（Tesla P100或其他）

## 安装指南

### 前置依赖

- C/C++编译工具链：gcc，g++，make，cmake
- CUDA相关工具：nvcc，nsight，compute
- Python相关工具：pixi，conda，pip

### 使用pixi安装（推荐）

本项目使用pixi作为环境管理工具，可以自动处理CUDA和Python依赖：

```bash
# 安装pixi（如果尚未安装）
curl -fsSL https://pixi.sh/install.sh | bash

# 克隆仓库
git clone https://github.com/loganautomata/cuda-dqn.git
cd cuda-dqn

# 创建虚拟环境并安装依赖
pixi install
```

### 手动安装

```bash
# 克隆仓库
git clone https://github.com/loganautomata/cuda-dqn.git
cd cuda-dqn

# 创建虚拟环境
conda create -n cuda-dqn python=3.8
conda activate cuda-dqn

# 安装PyTorch和CUDA工具包
conda install pytorch=1.10.0 torchvision=0.11.0 torchaudio=0.10.0 cudatoolkit=11.3 -c pytorch
```

## 使用方法

### 运行训练示例

```bash
nvcc -Xcompiler -fPIC -shared -o lib/qnet_cuda.so src/lib/qnet-latest.cu # 编译CUDA源码为共享库
python src/main.py # 运行训练
```

### 在自己的项目中使用CUDA加速的DQN

```python
# 加载cuda-dqn库
lib = ctypes.CDLL("./lib/qnet_cuda.so")
```

## 项目结构

```
cuda-dqn/
├── src/
│   ├── main.py                 # 训练入口
│   ├── algs/
│   │   ├── qlearner_cuda.py # DQN算法实现(CUDA版本)
│   │   └── qlearner_baseline.py # DQN算法实现(CPU版本)
│   ├── lib/
│   │   ├── qnet-latest.cu      # CUDA核函数实现
│   │   └── ...
│   └── utils/
│       ├── misc.py             # 辅助函数
│       ├── replay.py           # 经验回放缓冲区
│       └── ...
├── result/                     # 训练结果
│   └── ...
├── pixi.toml                   # pixi环境配置
└── README.md
```

## 许可证

本项目采用GPL-v3许可证 - 详情请参阅 LICENSE 文件

项目链接: [https://github.com/loganautomata/cuda-dqn](https://github.com/loganautomata/cuda-dqn)

Copyright (c) 2025 by www.logan.ren, All Rights Reserved.
