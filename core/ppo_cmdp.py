import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim):
        super(ActorCritic, self).__init__()
        # 🟢 State: [E_t, B_avail, C_remain] (dim=3)
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # 🟢 [핵심 수정] 연속형 액션 a_t (캐싱 비율)를 위한 Mean과 LogStd
        self.actor_mean = nn.Linear(64, 1)
        self.actor_logstd = nn.Parameter(torch.zeros(1)) # 로그 표준편차 학습
        
        # Reward Critic
        self.reward_critic = nn.Linear(64, 1)
        # Cost Critic
        self.cost_critic = nn.Linear(64, 1)

    def forward(self, state):
        features = self.shared_net(state)
        
        # Sigmoid를 적용해 평균값을 0~1 사이로 강제
        mean = torch.sigmoid(self.actor_mean(features))
        std = torch.exp(self.actor_logstd).expand_as(mean)
        
        state_value = self.reward_critic(features)
        cost_value = self.cost_critic(features)
        
        return mean, std, state_value, cost_value

class PPO_CMDP_Agent:
    def __init__(self, state_dim=3, lr=3e-4, gamma=0.99, clip_ratio=0.2, cost_limit=10.0, device="cpu"):
        self.device = device
        self.policy = ActorCritic(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.cost_limit = cost_limit 
        
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std, state_val, cost_val = self.policy(state_tensor)
            
        dist = Normal(mean, std)
        action = dist.sample()
        
        # 액션을 0.0 ~ 1.0 사이로 클리핑 (환경에 던질 값)
        action_clipped = torch.clamp(action, 0.0, 1.0)
        
        # 로그 확률은 클리핑 전의 원본 액션으로 계산해야 수학적으로 맞음
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action_clipped.cpu().numpy()[0], log_prob.item(), state_val.item(), cost_val.item()

    def _compute_returns(self, rewards):
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        return torch.FloatTensor(returns).to(self.device)

    def update(self, rollouts):
        states = torch.FloatTensor(np.array(rollouts['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(rollouts['actions'])).unsqueeze(-1).to(self.device) # Continuous Action shape 맞춤
        old_log_probs = torch.FloatTensor(rollouts['log_probs']).to(self.device)
        
        returns = self._compute_returns(rollouts['rewards'])
        cost_returns = self._compute_returns(rollouts['costs'])
        
        reward_vals = torch.FloatTensor(rollouts['reward_vals']).to(self.device)
        cost_vals = torch.FloatTensor(rollouts['cost_vals']).to(self.device)
        
        advantages = returns - reward_vals
        cost_advantages = cost_returns - cost_vals
        
        # Advantage 정규화 (학습 안정성)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(4): # K-epoch
            mean, std, curr_reward_vals, curr_cost_vals = self.policy(states)
            dist = Normal(mean, std)
            curr_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            ratios = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
            
            loss_cost_actor = (ratios * cost_advantages).mean()
            
            loss_reward_critic = F.mse_loss(curr_reward_vals.squeeze(), returns)
            loss_cost_critic = F.mse_loss(curr_cost_vals.squeeze(), cost_returns)

            # --- Löwdin 직교화 그래디언트 투영 (원본 유지) ---
            self.optimizer.zero_grad()
            
            loss_actor.backward(retain_graph=True)
            g_obj = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.policy.actor_mean.parameters()]
            
            self.optimizer.zero_grad()
            
            loss_cost_actor.backward(retain_graph=True)
            g_cost = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.policy.actor_mean.parameters()]
            
            self.optimizer.zero_grad()
            
            avg_cost = np.mean(rollouts['costs'])
            
            # 두 그래디언트 직교화 조작 연산 (actor_mean 레이어에만 적용)
            for p, g_o, g_c in zip(self.policy.actor_mean.parameters(), g_obj, g_cost):
                if avg_cost > self.cost_limit:
                    dot_product = torch.sum(g_o * g_c)
                    cost_norm_sq = torch.sum(g_c * g_c) + 1e-8
                    projection = (dot_product / cost_norm_sq) * g_c
                    p.grad = g_o - projection 
                else:
                    p.grad = g_o 
            
            # Critic 그래디언트 계산 및 전체 업데이트
            (loss_reward_critic + loss_cost_critic).backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()