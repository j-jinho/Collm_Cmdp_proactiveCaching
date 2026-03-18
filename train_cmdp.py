import numpy as np
import torch
from core.cmdp_env import CoLLM_CMDP_Env
from core.ppo_cmdp import PPO_CMDP_Agent
import matplotlib.pyplot as plt # 🟢 그래프 그리기용 라이브러리 추가

def train():
    print("🚀 CoLLM-CMDP 본격적인 학습을 시작합니다!")
    
    # 1. 환경 및 에이전트 초기화
    env = CoLLM_CMDP_Env(vocab_size=1000, device="cpu")
    
    # 상태 차원(3), 행동 차원(1000개 콘텐츠 ID), 비용 제한(C=10.0)
    agent = PPO_CMDP_Agent(
        state_dim=3, action_dim=1000, lr=3e-4, 
        gamma=0.99, clip_ratio=0.2, cost_limit=10.0, device="cpu"
    )
    
    max_episodes = 20 # 테스트를 위해 20번의 에피소드만 먼저 돌려봅니다
    
    history_reward = [] 
    history_cost = []

    # 2. 메인 학습 루프
    for ep in range(1, max_episodes + 1):
        obs, info = env.reset()
        
        # 궤적(Trajectory) 데이터를 저장할 딕셔너리
        rollouts = {
            'states': [], 'actions': [], 'log_probs': [], 
            'rewards': [], 'costs': [], 'reward_vals': [], 'cost_vals': []
        }
        
        ep_reward = 0.0
        ep_cost = 0.0
        step_count = 0
        
        while True:
            # 에이전트가 현재 상태(obs)를 보고 행동 선택 및 가치 평가
            action, log_prob, state_val, cost_val = agent.select_action(obs)
            
            # 선택한 행동을 ns-3 환경에 전달하고 결과(다음 상태, 보상, 비용 등) 받기
            next_obs, reward, terminated, truncated, info = env.step(action)
            cost = info.get('cost', 0.0)
            
            # 수집된 데이터 기록
            rollouts['states'].append(obs)
            rollouts['actions'].append(action)
            rollouts['log_probs'].append(log_prob)
            rollouts['rewards'].append(reward)
            rollouts['costs'].append(cost)
            rollouts['reward_vals'].append(state_val)
            rollouts['cost_vals'].append(cost_val)
            
            obs = next_obs
            ep_reward += reward
            ep_cost += cost
            step_count += 1
            
            # 시뮬레이션 종료 조건 (ns-3 시간이 다 되었을 때)
            if terminated or truncated:
                break
                
        # 3. 에피소드 종료 후 수집된 데이터로 에이전트 신경망 업데이트 (Löwdin 직교화 적용)
        agent.update(rollouts)
        
        # 학습 진행 상황 출력
        print(f"📈 Episode {ep}/{max_episodes} | Steps: {step_count} | Total Reward (Hit Rate): {ep_reward:.4f} | Total Cost (Penalties): {ep_cost:.4f}")

        # =========================================================
        # 🟢 바로 여기! for문 안쪽에 들여쓰기를 맞춰서 값들을 추가해 줍니다.
        history_reward.append(ep_reward)
        history_cost.append(ep_cost)
        # =========================================================
        
    env.close()
    # =================================================================
    # 🟢 1번: 학습 종료 후 시각화 그래프 이미지(PNG) 저장
    # =================================================================
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_episodes + 1), history_reward, label='Total Reward (Hit Rate)', color='blue', marker='o')
    plt.plot(range(1, max_episodes + 1), history_cost, label='Total Cost (Constraint)', color='red', marker='x')
    plt.axhline(y=10.0, color='r', linestyle='--', label='Cost Limit (C=10.0)') # 제약 조건 선
    plt.title('PPO-CMDP Learning Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png') # 이미지 파일로 저장
    print("📊 학습 그래프가 'learning_curve.png'로 저장되었습니다!")
    print("✅ 모든 학습이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    train()