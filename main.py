import random
import gym
import numpy as np
import torch

from algs.DQN.torch.DQN import DQN as DQN_pytorch
from algs.DQN.cuda.DQN import DQN as DQN_cuda
from algs.A2C.torch.A2C import A2C as A2C_pytorch
from algs.A2C.cuda.A2C import A2C as A2C_cuda

def setRandomSeed(env):
    random.seed(0)
    np.random.seed(0)
    env.reset(seed=0)
    torch.manual_seed(0)
    
def main():
    
    model_type = 2
    device_type = 0
    
    cpu_device = torch.device('cpu')
    gpu_device = torch.device("cuda")

    # 选择训练任务为 CartPole 类型
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    
    setRandomSeed(env)
    
    # 连续状态空间：CartPole的状态维度 = 4
    state_dim = env.observation_space.shape[0]
    # 动作参数获取, CartPole 的动作空间为 2（左 / 右）
    action_dim = env.action_space.n
    # a = 4, c = 2

    # A2C
    # pytorch gpu
    # 500 train executed in 17201.75 ms
    if model_type == 1 and device_type==0:
        A2C_pytorch(state_dim, action_dim, gpu_device, env).train()
    # pytorch cpu
    # 500 train executed in 9336.46 ms
    elif model_type == 1 and device_type == 1:
        A2C_pytorch(state_dim, action_dim, cpu_device, env).train()
    # cuda gpu
    # 500 train executed in 17737.08 ms
    elif model_type == 2 and device_type==0:
        A2C_cuda(state_dim, action_dim, gpu_device, env).train()

    # pytorch gpu
    # 100 train executed in 1680.97 ms
    # 500 train executed in 369747.02 ms
    elif model_type == 3 and device_type==0:
        DQN_pytorch(state_dim, action_dim, gpu_device, env).train()
    # pytorch cpu
    # 100 train executed in 653.07 ms
    # 500 train executed in 120614.12 ms
    elif model_type == 3 and device_type==1:
        DQN_pytorch(state_dim, action_dim, cpu_device, env).train()
    # cuda gpu
    # 100 train executed in 18613.35 ms
    else:
        DQN_cuda(state_dim, action_dim, gpu_device, env).train()


if __name__ == "__main__":
    main()


