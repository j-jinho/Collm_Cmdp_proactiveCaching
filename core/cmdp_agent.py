import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.costs = []
        self.is_terminals = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.costs[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Actor network (행동 결정망)
        self.actor_fc1 = nn.Linear(state_dim, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_mean = nn.Linear(64, action_dim)
        # 행동의 분산(탐험 정도)을 학습하는 파라미터
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic network (상태 가치 평가망)
        self.critic_fc1 = nn.Linear(state_dim, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_value = nn.Linear(64, 1)

    def forward(self):
        raise NotImplementedError
        
    def act(self, state, deterministic=False):
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        
        # Caching ratio a_t는 0.0 ~ 1.0 사이어야 하므로 Sigmoid 적용
        action_mean = torch.sigmoid(self.actor_mean(x)) 
        
        # Test 모드일 때는 탐험 없이 가장 좋은(확정적인) 행동만 선택
        if deterministic:
            return action_mean.detach(), None
            
        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        
        # [0, 1] 범위로 클리핑하여 안전한 캐싱 비율 보장
        action = torch.clamp(action, 0.0, 1.0)
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
        
    def evaluate(self, state, action):
        # Actor 평가
        x_a = F.relu(self.actor_fc1(state))
        x_a = F.relu(self.actor_fc2(x_a))
        action_mean = torch.sigmoid(self.actor_mean(x_a))
        action_std = torch.exp(self.actor_log_std)
        
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        # Critic 평가
        x_c = F.relu(self.critic_fc1(state))
        x_c = F.relu(self.critic_fc2(x_c))
        state_values = self.critic_value(x_c)
        
        return action_logprobs, state_values, dist_entropy


class PPOLagrangianAgent:
    # =========================================================
    # 🟢 [핵심] Optuna에서 넣어주는 하이퍼파라미터 변수들을 여기서 받습니다!
    # =========================================================
    def __init__(self, state_dim=3, action_dim=1, 
                 lr_actor=3e-4, lr_critic=1e-3, lr_lambda=5e-3, gamma=0.99,
                 K_epochs=10, eps_clip=0.2, cost_limit=0.0):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.cost_limit = cost_limit
        
        self.memory = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim)
        
        # Actor와 Critic에 각기 다른 학습률(lr) 적용
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor_fc1.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_fc2.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
            {'params': [self.policy.actor_log_std], 'lr': lr_actor},
            {'params': self.policy.critic_fc1.parameters(), 'lr': lr_critic},
            {'params': self.policy.critic_fc2.parameters(), 'lr': lr_critic},
            {'params': self.policy.critic_value.parameters(), 'lr': lr_critic}
        ])
        
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        # Lagrangian Multiplier 세팅
        self.lambda_val = 0.0
        self.lr_lambda = lr_lambda

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, action_logprob = self.policy_old.act(state, deterministic)
        
        self.memory.states.append(state)
        self.memory.actions.append(action)
        if action_logprob is not None:
            self.memory.logprobs.append(action_logprob)
            
        return action.item()

    def update_lagrangian(self, episode_cost):
        # 네트워크 위반 점수(Cost)에 따라 채찍질 강도(Lambda) 업데이트
        self.lambda_val = max(0.0, self.lambda_val + self.lr_lambda * (episode_cost - self.cost_limit))

    def train_step(self):
        if len(self.memory.states) == 0:
            return

        # 🟢 [CMDP 핵심 논리] 보상(Reward)에서 위반 페널티(lambda * cost)를 차감!
        combined_rewards = []
        discounted_reward = 0
        
        for reward, cost, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.costs), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            
            # 여기서 에이전트의 뇌 구조가 완성됨: "위반하면 보상을 깎는다!"
            penalized_reward = reward - (self.lambda_val * cost)
            discounted_reward = penalized_reward + (self.gamma * discounted_reward)
            combined_rewards.insert(0, discounted_reward)
            
        # 보상 정규화 (안정적인 학습을 위해)
        combined_rewards = torch.tensor(combined_rewards, dtype=torch.float32)
        if len(combined_rewards) > 1:
            combined_rewards = (combined_rewards - combined_rewards.mean()) / (combined_rewards.std() + 1e-7)
        
        # =========================================================
        # 🟢 [에러 원인 완벽 해결] 중복 저장된 찌꺼기 필터링 로직!
        # main.py에서 들어온 모양이 다른 데이터를 무시하고, 
        # select_action()에서 정상 저장한 [1, X] 형태의 텐서만 골라냅니다.
        # =========================================================
        valid_states = [s for s in self.memory.states if isinstance(s, torch.Tensor) and s.dim() == 2]
        valid_actions = [a for a in self.memory.actions if isinstance(a, torch.Tensor) and a.dim() == 2]
        valid_logprobs = [lp for lp in self.memory.logprobs if isinstance(lp, torch.Tensor)]
        
        old_states = torch.squeeze(torch.stack(valid_states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(valid_actions, dim=0)).detach().unsqueeze(1)
        old_logprobs = torch.squeeze(torch.stack(valid_logprobs, dim=0)).detach().unsqueeze(1)
        
        # K_epochs 만큼 PPO 알고리즘 업데이트 반복
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = combined_rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # 최종 손실 함수 계산 (Actor Loss + Critic Loss - Entropy Bonus)
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, combined_rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()