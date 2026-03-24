import argparse 
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt

# 💡 텐서보드 기록용 라이브러리 추가!
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import CoLLMDataLoader
from core.collm_engine import OriginalCoLLMEngine
from core.cmdp_agent import PPOLagrangianAgent
from envs.ns3_interface import EdgeCachingCMDPEnv 
from train_sasrec import LongTermPreferenceEncoder 

def main():
    parser = argparse.ArgumentParser(description="CoLLM-CMDP Edge Caching Simulation")
    parser.add_argument('--mode', type=str, default='collm_cmdp', 
                        choices=['collm_cmdp', 'ppo_only', 'pure_reactive', 'fixed_proactive'], 
                        help='실험할 알고리즘 모드를 선택하세요.')
    args = parser.parse_args()

    print(f"🚀 [{args.mode.upper()}] 모드로 [리얼 데이터 동기화 완료] 엣지 캐싱 본 실험을 시작합니다...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    tb_log_dir = os.path.join(current_dir, "runs", f"Experiment_{args.mode}")
    os.makedirs(tb_log_dir, exist_ok=True) 
    
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"📊 TensorBoard 로깅 시작! 터미널에 'tensorboard --logdir {tb_log_dir}'을 입력하세요.")

    # --- Phase 1: 데이터 로더 로드 ---
    dataloader = CoLLMDataLoader(raw_dir="data/raw")

    print("🧠 [1/2] CoLLM (LLM) 엔진을 VRAM(GPU)에 적재합니다...")
    collm = OriginalCoLLMEngine(device=device, sasrec_dim=128)
    cie_weight_path = "core/weights/trained_cie.pth"
    if os.path.exists(cie_weight_path):
        collm.cie.load_state_dict(torch.load(cie_weight_path, map_location=device))
    collm.eval() 

    # 🟢 [WSL OOM 영구 해결] 인코더를 GPU에 올리지 않고 CPU RAM에 둡니다!
    print("🧠 [2/2] SASRec 인코더(128d)를 CPU RAM에 적재합니다 (WSL VRAM 버그 회피)...")
    idx2item_path = 'core/weights/idx2item_map.pth'
    encoder_path = 'core/weights/long_term_encoder.pth'
    idx2item = torch.load(idx2item_path)
    item_count = len(idx2item)
    
    item2idx = {item: idx for idx, item in idx2item.items()}
    
    encoder = LongTermPreferenceEncoder(item_count=item_count, hidden_dim=128)
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    encoder.eval() 
        
    # --- 환경 및 에이전트 초기화 ---
    agent = PPOLagrangianAgent(state_dim=3, action_dim=1)
    env = EdgeCachingCMDPEnv(target_name="edge_caching_sim") 

    MAX_STEPS_PER_DAY = 1440 
    TRAIN_DAYS = 14
    TEST_DAYS = 7
    TOTAL_DAYS = TRAIN_DAYS + TEST_DAYS
    
    history_reward, history_cost = [], []

    for day in range(1, TOTAL_DAYS + 1):
        is_train = day <= TRAIN_DAYS 
        phase_name = "TRAIN" if is_train else "TEST"
        
        ns3_state, info = env.reset()
        c_remain = ns3_state[2] if len(ns3_state) > 2 else 1.0 
        
        day_reward, day_cost, day_cost_scaled = 0.0, 0.0, 0.0
        
        ep_et_list, ep_at_list = [], []
        
        for step in range(MAX_STEPS_PER_DAY):
            absolute_step = (day - 1) * MAX_STEPS_PER_DAY + step
            
            traffic_info = dataloader.get_real_traffic_at_t(absolute_step)
            b_avail = traffic_info['b_avail']
            pred_domain = traffic_info['domain']
            real_req_id = traffic_info['item_id'] 
            user_id = traffic_info['user_id']
            
            # 텐서도 .to(device) 없이 CPU에 유지
            real_seq_tensor = dataloader.get_user_seq_tensor(user_id, real_req_id, item2idx, max_len=50)
            
            with torch.no_grad():
                # 추천 벡터 연산을 CPU에서 처리
                vector_128d = encoder(real_seq_tensor)
                
                if hasattr(encoder, 'item_emb'):
                    scores = torch.matmul(vector_128d, encoder.item_emb.weight.T)
                    pred_id = torch.argmax(scores, dim=-1).item()
                else:
                    pred_id = 1

            recent_hist = dataloader.get_dynamic_recent_domains(user_id, pred_domain)
            recent_10_str = f"[{', '.join(recent_hist)}]"
            top_k_str = f"[{pred_domain}, SHORT]"
            
            pred_type, e_t = collm.forward_and_get_uncertainty(vector_128d, recent_10_str, top_k_str)
            
            if np.isnan(e_t): e_t = 0.5
            if np.isnan(b_avail): b_avail = 100.0
            if np.isnan(c_remain): c_remain = 0.5
            
            b_avail_norm = np.clip(b_avail / 250.0, 0.0, 1.0)
            state = np.array([e_t, b_avail_norm, c_remain], dtype=np.float32)
            
            if args.mode == 'pure_reactive':
                a_t = 0.0  
            elif args.mode == 'fixed_proactive':
                a_t = 0.3  
            else:
                try:
                    a_t = agent.select_action(state, deterministic=not is_train) 
                except TypeError:
                    a_t = agent.select_action(state)
            
            next_state, reward, done, truncated, info = env.step(
                action=[a_t], predicted_traffic_type=pred_type, b_avail=b_avail, 
                e_t=e_t, pred_id=pred_id, real_req_id=real_req_id
            )
            
            cost = info.get("cost", 0.0) 
            cost_scaled = cost / 50.0
            
            if is_train and args.mode in ['collm_cmdp', 'ppo_only']:
                agent.memory.rewards.append(reward)
                agent.memory.costs.append(cost_scaled) 
                agent.memory.is_terminals.append(done)
            
            day_reward += reward
            day_cost += cost               
            day_cost_scaled += cost_scaled 
            c_remain = next_state[2]
            
            ep_et_list.append(e_t)
            ep_at_list.append(a_t)
            
            if step % 200 == 0:
                print(f"[{phase_name}] Day {day:2d}| Step {step:4d} | E_t: {e_t:.2f} | B_avail: {b_avail:.1f} | Action(a_t): {a_t:.2f} | HitRate: {reward:.3f}")
                
            if done or truncated: break
                
        if is_train and args.mode in ['collm_cmdp', 'ppo_only']:
            agent.train_step()
            if args.mode == 'collm_cmdp':
                agent.update_lagrangian(day_cost_scaled)
        
        history_reward.append(day_reward)
        history_cost.append(day_cost) 
        
        avg_et = np.mean(ep_et_list)
        avg_at = np.mean(ep_at_list)
        
        print(f"🏁 {phase_name} Day {day} 완료 | 총 보상: {day_reward:.2f} | 총 위반: {day_cost:.2f} | 람다: {agent.lambda_val:.4f}\n")

        writer.add_scalar(f"1. Performance/Total_Reward ({phase_name})", day_reward, day)
        writer.add_scalar(f"1. Performance/Total_Cost ({phase_name})", day_cost, day)
        writer.add_scalar(f"2. Agent_Dynamics/Average_Action ({phase_name})", avg_at, day)
        if args.mode == 'collm_cmdp':
            writer.add_scalar(f"2. Agent_Dynamics/Lagrangian_Lambda ({phase_name})", agent.lambda_val, day)
        writer.add_scalar(f"3. Environment/Average_Uncertainty ({phase_name})", avg_et, day)

    env.close()
    writer.close()

    # =========================================================
    # 🟢 [수정] 이중 Y축 적용하여 Reward와 Cost를 시각적으로 분리!
    # =========================================================
    fig, ax1 = plt.subplots(figsize=(12, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Total Reward (Hit Rate)', color=color1)
    ax1.plot(range(1, TOTAL_DAYS + 1), history_reward, label='Total Reward', color=color1, marker='o')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('Total Cost (Violation)', color=color2)  
    ax2.plot(range(1, TOTAL_DAYS + 1), history_cost, label='Total Cost', color=color2, marker='x')
    ax2.axhline(y=0.0, color='r', linestyle='--', label='Cost Limit (0.0)')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.axvline(x=TRAIN_DAYS + 0.5, color='gray', linestyle='--', label='Train/Test Split') 
    plt.title(f'[{args.mode.upper()}] Learning Curve (14-Day Train, 7-Day Test)')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()  
    plt.grid(True)
    plt.savefig('learning_curve.png')

    last_ep_states = np.array(agent.memory.states)
    last_ep_actions = np.array(agent.memory.actions)
    
    if len(last_ep_states) > 0:
        e_t_history = last_ep_states[:, 0]
        b_avail_history = last_ep_states[:, 1] 
        a_t_history = last_ep_actions
        
        plt.figure(figsize=(12, 6))
        plt.plot(e_t_history, label='CoLLM Uncertainty (E_t)', linestyle='--', color='orange')
        plt.plot(b_avail_history, label='Normalized Bandwidth (B_avail)', linestyle='-.', color='green')
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