import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Beta

class PPOMemory:
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.costs, self.is_terminals = [], [], []
        self.state_values, self.cost_values = [], []

    def clear(self):
        del self.states[:], self.actions[:], self.logprobs[:]
        del self.rewards[:], self.costs[:], self.is_terminals[:]
        del self.state_values[:], self.cost_values[:]

class PPOLagrangianAgent:
    def __init__(self, state_dim=3, action_dim=1, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        self.actor_alpha = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, action_dim), nn.Softplus())
        self.actor_beta = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, action_dim), nn.Softplus())
        
        self.critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.cost_critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        
        self.optimizer = optim.Adam([
            {'params': self.actor_alpha.parameters(), 'lr': lr},
            {'params': self.actor_beta.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr},
            {'params': self.cost_critic.parameters(), 'lr': lr}
        ])
        
        self.memory = PPOMemory()
        self.lambda_val = 0.0 
        self.lambda_lr = 0.01

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            alpha = self.actor_alpha(state_tensor) + 1.0 
            beta = self.actor_beta(state_tensor) + 1.0
            
            dist = Beta(alpha, beta)
            action = dist.sample() 
            
            self.memory.states.append(state)
            self.memory.actions.append(action.numpy()[0])
            self.memory.logprobs.append(dist.log_prob(action).numpy()[0])
            self.memory.state_values.append(self.critic(state_tensor).numpy()[0])
            self.memory.cost_values.append(self.cost_critic(state_tensor).numpy()[0])
            
        return action.numpy()[0][0]

    def update_lagrangian(self, epoch_cost):
        cost_limit = 0.0 
        self.lambda_val = max(0.0, self.lambda_val + self.lambda_lr * (epoch_cost - cost_limit))
        print(f"🔄 Lagrangian Multiplier Updated: {self.lambda_val:.4f}")
        
    def train_step(self):
        rewards, costs = [], []
        discounted_r, discounted_c = 0, 0
        for r, c, is_term in zip(reversed(self.memory.rewards), reversed(self.memory.costs), reversed(self.memory.is_terminals)):
            if is_term: discounted_r, discounted_c = 0, 0
            discounted_r = r + (self.gamma * discounted_r)
            discounted_c = c + (self.gamma * discounted_c)
            rewards.insert(0, discounted_r)
            costs.insert(0, discounted_c)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        costs = torch.tensor(costs, dtype=torch.float32)
        
        old_states = torch.FloatTensor(np.array(self.memory.states))
        old_actions = torch.FloatTensor(np.array(self.memory.actions))
        old_logprobs = torch.FloatTensor(np.array(self.memory.logprobs))
        old_state_values = torch.FloatTensor(np.array(self.memory.state_values)).squeeze()
        old_cost_values = torch.FloatTensor(np.array(self.memory.cost_values)).squeeze()

        advantages = rewards - old_state_values.detach()
        cost_advantages = costs - old_cost_values.detach()
        lagrangian_advantages = advantages - (self.lambda_val * cost_advantages)

        for _ in range(4): 
            alpha = self.actor_alpha(old_states) + 1.0
            beta = self.actor_beta(old_states) + 1.0
            dist = Beta(alpha, beta)
            
            ratios = torch.exp(dist.log_prob(old_actions) - old_logprobs.detach())
            surr1 = ratios * lagrangian_advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * lagrangian_advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(self.critic(old_states).squeeze(), rewards)
            cost_critic_loss = nn.MSELoss()(self.cost_critic(old_states).squeeze(), costs)

            loss = actor_loss + 0.5 * critic_loss + 0.5 * cost_critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.memory.clear()