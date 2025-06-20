from utils.misc import timeit
from .ActorCritic import ActorCritic
import matplotlib.pyplot as plt
import torch
import numpy as np

from .AdamOptimizer import AdamOptimizer


class A2C:
    
    __hidden_size = 256  # 隐藏层大小
    __lr          = 1e-3 # 学习率
    __num_steps   = 10
    __max_frames   = 500  # 总训练次数

    
    def _compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value[0]
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def _plot(self, frame_idx, rewards):
        plt.plot(rewards,'b-')
        plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
        plt.pause(0.0001)
        
    # 测试当前模型的效果
    def _test_env(self, vis=False):
        state = self.env.reset()[0]
        if vis: self.env.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _, _ = self.model.forward(state.tolist())
            next_state, reward, done, _ , _ = self.env.step(dist.sample().unsqueeze(0).cpu().numpy()[0])
            state = next_state
            if vis: self.env.render()
            total_reward += reward
        return total_reward
    
    def __init__(self, state_dim, action_dim, device, env):
        self.device = device
        self.env = env
        self.model = ActorCritic(state_dim, action_dim , self.__hidden_size).to(device)
        self.optimizer = AdamOptimizer(self.model.parameters(), lr=self.__lr)

    @timeit
    def train(self):

        # 开启交互式画图
        plt.ion()
        
        frame_idx    = 0
        test_rewards = []   # 测试奖励
        state = self.env.reset()[0]
        
        while frame_idx < self.__max_frames: 
            states = []
            probs = []
            actions = []
            log_probs = []  # 每一步的对数概率
            values    = []  # 每一步的预估值
            rewards   = []  # 每一步的奖励
            masks     = []
            entropy = 0     # 熵

            # rollout trajectory
            # 每轮收集num_steps步的交互数据
            for _ in range(self.__num_steps):
                # 当前状态
                state = torch.FloatTensor(state).to(self.device)
                states.append(state.tolist())
                # 获得动作概率和值
                dist, value, prob = self.model.forward(state)
                probs.append(prob.tolist())

                # 预期动作
                action = dist.sample()
                actions.append(action)
                
                # 执行动作
                next_state, reward, done, _ , _ = self.env.step(action.cpu().numpy())

                # 对数概率
                log_prob = dist.log_prob(action)
                # 熵
                entropy += dist.entropy().mean()
                
                log_probs.append(log_prob.unsqueeze(0))
                values.append(value)
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(self.device))
                
                # 获取下一个状态
                state = next_state
                frame_idx += 1
                
                # 每10次训练测试一下当前模型的效果
                if frame_idx % 10 == 0:
                    # 测试10次，取平均结果
                    test_rewards.append(np.mean([self._test_env() for _ in range(10)]))
                    self._plot(frame_idx, test_rewards)
                    
            # 输入下一状态
            next_state = torch.FloatTensor(next_state)
            # 获取下一状态的价值估计
            _, next_value, _ = self.model.forward(next_state.tolist())
            # 计算折扣累积回报
            returns = self._compute_returns(next_value, rewards, masks)
            
            log_probs = torch.cat(log_probs).to(self.device).view(-1, 1)
            returns   = torch.cat(returns).detach()
            values    = [torch.tensor(v, dtype=torch.float32) if isinstance(v, list) else v for v in values]
            values    = torch.cat(values).to(self.device).view(-1, 1)

            # 优势函数：回报与价值估计的差值
            advantage = returns - values


            # 策略网络损失（最大化期望回报）
            actor_loss  = -(log_probs * advantage.detach()) # 10 * 1
            actor_loss_mean  = actor_loss.mean()
            # 价值网络损失（最小化价值估计误差）
            critic_loss = advantage.pow(2)  # 10 * 1
            critic_loss_mean = critic_loss.mean()
            # 总损失：策略损失 + 价值损失 - 熵正则化（鼓励探索）
            loss = actor_loss_mean + 0.5 * critic_loss_mean - 0.001 * entropy

            actor_input = states  # 10 * 4
            actor_output = probs  # 10 * 2


            # 假设actor输出是动作概率分布，维度为 [batch_size, action_dim]
            batch_size = len(probs)
            action_dim = len(probs[0])

            # 创建actor输出层梯度矩阵 [batch_size, action_dim]
            actor_grad = torch.zeros((batch_size, action_dim), device=self.device)

            # 对每个样本，只有选中动作的梯度非零
            for i in range(batch_size):
                action = actions[i].item()
                # 策略梯度: ∂Loss/∂log_prob = -advantage
                # ∂Loss/∂prob = ∂Loss/∂log_prob * ∂log_prob/∂prob = -advantage * (1/prob)
                # 简化为: -advantage[i] * (1/probs[i, action])
                actor_grad[i, action] = -advantage[i].item() / (probs[i][action] + 1e-10)
            actor_grad = actor_grad.tolist()

            # 计算critic网络输出层梯度
            # ∂Loss/∂value = 2 * (value - target)
            critic_grad = 2 * advantage
            critic_grad = critic_grad.squeeze().tolist()

            max_grad_norm = 1.0
            actor_grad = np.clip(actor_grad, -max_grad_norm, max_grad_norm)
            actor_grad = actor_grad.tolist()


            # 梯度更新
            self.model.zero_grad()
            self.model.backward(actor_input, actor_output, actor_grad, critic_grad)
            self.optimizer.step(self.model.gradients(), self.model.parameters())

        # 保存图形
        plt.savefig('result/A2C.png', dpi=300, bbox_inches='tight')
        plt.close() 

        