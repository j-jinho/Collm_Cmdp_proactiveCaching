import numpy as np
import torch
import pickle
import random
import os
import matplotlib.pyplot as plt

# 💡 텐서보드 기록용 라이브러리 추가!
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import CoLLMDataLoader
from core.collm_engine import OriginalCoLLMEngine
from core.cmdp_agent import PPOLagrangianAgent
from envs.ns3_interface import EdgeCachingCMDPEnv 
from train_sasrec import LongTermPreferenceEncoder 

def main():
    print("🚀 CoLLM-CMDP [리얼 데이터 동기화 완료] 엣지 캐싱 본 실험을 시작합니다...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================
    # 📊 TensorBoard 초기화 (runs/CoLLM_CMDP_Experiment 폴더에 저장)
    # =========================================================
    tb_log_dir = "runs/CoLLM_CMDP_Experiment"
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"📊 TensorBoard 로깅 시작! 터미널에 'tensorboard --logdir {tb_log_dir}'을 입력하세요.")

    # --- Phase 2: 128d 인코더 로드 ---
    idx2item_path = 'core/weights/idx2item_map.pth'
    encoder_path = 'core/weights/long_term_encoder.pth'
    idx2item = torch.load(idx2item_path)
    item_count = len(idx2item)
    
    encoder = LongTermPreferenceEncoder(item_count=item_count, hidden_dim=128).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()

    # --- Phase 3: CoLLM 엔진 로드 ---
    dataloader = CoLLMDataLoader(raw_dir="data/raw")
    collm = OriginalCoLLMEngine(device=device, sasrec_dim=128)
    cie_weight_path = "core/weights/trained_cie.pth"
    if os.path.exists(cie_weight_path):
        collm.cie.load_state_dict(torch.load(cie_weight_path, map_location=device))
    collm.eval() 
        
    # --- 환경 및 에이전트 초기화 ---
    agent = PPOLagrangianAgent(state_dim=3, action_dim=1)
    env = EdgeCachingCMDPEnv(target_name="edge_caching_sim") 

    MAX_STEPS = 1440 
    EPISODES = 30 
    history_reward, history_cost = [], []

    for episode in range(1, EPISODES + 1):
        ns3_state, info = env.reset()
        c_remain = ns3_state[2] if len(ns3_state) > 2 else 1.0 
        episode_reward, episode_cost = 0.0, 0.0 
        
        # 분석용 임시 변수
        ep_et_list, ep_at_list = [], []
        
        for step in range(MAX_STEPS):
            traffic_info = dataloader.get_real_traffic_at_t(step)
            b_avail = traffic_info['b_avail']
            pred_domain = traffic_info['domain']
            pred_id = traffic_info['item_id'] 
            
            # 더미 시퀀스로 128d 추출
            dummy_seq = torch.randint(1, item_count + 1, (1, 50), device=device)
            with torch.no_grad():
                vector_128d = encoder(dummy_seq)

            recent_hist = [random.choice(["NEWS", "SHORT", "VOD"]) for _ in range(3)]
            recent_10_str = f"[{', '.join(recent_hist)}]"
            top_k_str = f"[{pred_domain}, SHORT]"
            
            pred_type, e_t = collm.forward_and_get_uncertainty(vector_128d, recent_10_str, top_k_str)
            
            if np.isnan(e_t): e_t = 0.5
            if np.isnan(b_avail): b_avail = 100.0
            if np.isnan(c_remain): c_remain = 0.5
            
            state = np.array([e_t, b_avail, c_remain], dtype=np.float32)
            a_t = agent.select_action(state) 
            
            next_state, reward, done, truncated, info = env.step(
                action=[a_t], predicted_traffic_type=pred_type, b_avail=b_avail, e_t=e_t, pred_id=pred_id
            )
            
            cost = info.get("cost", 0.0) 
            agent.memory.rewards.append(reward)
            agent.memory.costs.append(cost)
            agent.memory.is_terminals.append(done)
            
            episode_reward += reward
            episode_cost += cost
            c_remain = next_state[2]
            
            ep_et_list.append(e_t)
            ep_at_list.append(a_t)
            
            if step % 200 == 0:
                print(f"Ep {episode}| Step {step:4d} | E_t: {e_t:.2f} | B_avail: {b_avail:.1f} | Action(a_t): {a_t:.2f} | HitRate: {reward:.3f}")
                
            if done or truncated: break
                
        agent.train_step()
        agent.update_lagrangian(episode_cost)
        
        history_reward.append(episode_reward)
        history_cost.append(episode_cost)
        
        avg_et = np.mean(ep_et_list)
        avg_at = np.mean(ep_at_list)
        
        print(f"🏁 Episode {episode} 완료 | 총 보상: {episode_reward:.2f} | 총 위반: {episode_cost:.2f} | 람다: {agent.lambda_val:.4f}\n")

        # =========================================================
        # 📊 TensorBoard에 에피소드 결과 실시간 기록!
        # =========================================================
        writer.add_scalar("1. Performance/Total_Reward (Hit Rate)", episode_reward, episode)
        writer.add_scalar("1. Performance/Total_Cost (Violation)", episode_cost, episode)
        writer.add_scalar("2. Agent_Dynamics/Lagrangian_Multiplier (Lambda)", agent.lambda_val, episode)
        writer.add_scalar("2. Agent_Dynamics/Average_Action (a_t)", avg_at, episode)
        writer.add_scalar("3. Environment/Average_Uncertainty (E_t)", avg_et, episode)

    env.close()
    writer.close() # 로깅 종료

    # (이하 기존 Matplotlib 시각화 파트 동일)
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, EPISODES + 1), history_reward, label='Total Reward (Hit Rate)', color='blue', marker='o')
    plt.plot(range(1, EPISODES + 1), history_cost, label='Total Cost (Violation)', color='red', marker='x')
    plt.axhline(y=0.0, color='r', linestyle='--', label='Cost Limit (0.0)')
    plt.title('CoLLM-CMDP Learning Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')

    last_ep_states = np.array(agent.memory.states)
    last_ep_actions = np.array(agent.memory.actions)
    
    if len(last_ep_states) > 0:
        e_t_history = last_ep_states[:, 0]
        b_avail_history = last_ep_states[:, 1] / 1000.0 
        a_t_history = last_ep_actions
        
        plt.figure(figsize=(12, 6))
        plt.plot(e_t_history, label='CoLLM Uncertainty (E_t)', linestyle='--', color='orange')
        plt.plot(b_avail_history, label='Bandwidth (B_avail / 1000)', linestyle='-.', color='green')
        plt.plot(a_t_history, label='Caching Ratio Action (a_t)', linewidth=2, color='blue')
        
        plt.title('Agent Decision Dynamics (Last 24h Episode)')
        plt.xlabel('Simulation Steps (Minutes)')
        plt.ylabel('Normalized Values (0.0 ~ 1.0)')
        plt.legend()
        plt.grid(True)
        plt.savefig('decision_dynamics.png')
        print("📊 시각화 결과물 'learning_curve.png', 'decision_dynamics.png' 저장 완료!")

if __name__ == "__main__":
    main()