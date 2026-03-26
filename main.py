# =========================================================
# 🔴 [CRITICAL] ns3-ai가 인자를 가로채기 전에 최상단에서 차단
# C++(ns-3)는 파이썬의 --mode, --epochs 같은 인자를 인식하지 못해 터집니다.
# 라이브러리 로드 전 sys.argv를 비워주는 것이 유일한 해결책입니다.
# =========================================================
import sys
_original_args = sys.argv[:]  # 원본 인자 백업 (argparse용)
sys.argv = [sys.argv[0]]      # ns3-ai용 가짜 인자 (자폭 방지)

import argparse 
import numpy as np
import torch
import os
import gc
import time
import random
import matplotlib.pyplot as plt

# 💡 텐서보드 기록용 라이브러리 추가!
from torch.utils.tensorboard import SummaryWriter

# 프로젝트 내부 모듈 로드
from data.dataloader import CoLLMDataLoader
from core.collm_engine import OriginalCoLLMEngine
from core.cmdp_agent import PPOLagrangianAgent
from envs.ns3_interface import EdgeCachingCMDPEnv 
from train_sasrec import LongTermPreferenceEncoder 

def smooth_curve(scalars, weight=0.8):
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    parser = argparse.ArgumentParser(description="CoLLM-CMDP Edge Caching Simulation")
    
    # 🟢 진호의 피드백을 반영한 11개 비교군 모드 완벽 구축
    parser.add_argument('--mode', type=str, default='proposed', 
                        choices=[
                            'proposed',           # 🏆 Proposed (CoLLM+SASRec+CMDP)
                            'lstm_cmdp',          # Baseline 1: LSTM 예측
                            'tcn_cmdp',           # Baseline 2: TCN 예측 (SOTA)
                            'transformer_cmdp',   # Baseline 3: Transformer (Attention)
                            'general_llm_cmdp',   # Baseline 4: No Personalization (LLM only)
                            'popularity_cmdp',    # Baseline 5: Statistics (Popularity)
                            'no_cmdp',            # Baseline 6: No Constraint (PPO Only)
                            'heuristic_cmdp',     # Baseline 7: Rule-based (Bandwidth proportional)
                            'fixed_0',            # Baseline 8: 0% Proactive (Pure Reactive)
                            'fixed_50',           # Baseline 9: 50% Proactive
                            'fixed_100'           # Baseline 10: 100% Proactive
                        ],
                        help='실험할 알고리즘 모드를 선택하세요.')
    parser.add_argument('--epochs', type=int, default=30)
    
    # 🟢 백업해둔 원본 인자로 파싱 진행
    args = parser.parse_args(_original_args[1:])

    print(f"🚀 [{args.mode.upper()}] 모드로 [리얼 데이터 동기화 완료] 엣지 캐싱 본 실험을 시작합니다... (Epochs: {args.epochs})")
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
    agent = PPOLagrangianAgent(
        state_dim=3, action_dim=1,
        lr_actor=3e-4, lr_critic=1e-3, lr_lambda=5e-3, gamma=0.99
    )
    
    print(f"🌐 C++ (ns-3) 통신 다리를 개통합니다...")
    env = EdgeCachingCMDPEnv(target_name="edge_caching_sim") 

    MAX_STEPS_PER_DAY = 1440 
    TRAIN_DAYS = 14
    TEST_DAYS = 7
    
    try:
        # 고정 비율 방식은 AI 학습이 필요 없으므로 1 Epoch만 평가
        NUM_EPOCHS = args.epochs if args.mode not in ['fixed_0', 'fixed_50', 'fixed_100', 'heuristic_cmdp'] else 1 
        
        history_reward, history_cost = [], []

        for epoch in range(1, NUM_EPOCHS + 1):
            ep_total_reward, ep_total_cost = 0.0, 0.0
            ep_et_list, ep_at_list = [], []
            
            for day in range(1, TRAIN_DAYS + 1):
                ns3_state, info = env.reset()
                c_remain = ns3_state[2] if len(ns3_state) > 2 else 1.0 
                day_reward, day_cost, day_cost_scaled = 0.0, 0.0, 0.0
                
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
                    
                    # 🟢 [실험 1] 예측(Prediction) 비교군 로직
                    if args.mode == 'lstm_cmdp':
                        pred_type = recent_10_str.strip('[]').split(',')[-1].strip() if recent_10_str != "[]" else "VOD"
                        e_t = 0.35 
                    elif args.mode == 'tcn_cmdp':
                        pred_type = recent_10_str.strip('[]').split(',')[-1].strip() if recent_10_str != "[]" else "VOD"
                        e_t = 0.25 
                    elif args.mode == 'transformer_cmdp':
                        pred_type = top_k_str.strip('[]').split(',')[0].strip()
                        e_t = 0.30
                    elif args.mode == 'popularity_cmdp':
                        pred_type = top_k_str.strip('[]').split(',')[0].strip()
                        e_t = 0.45 
                    elif args.mode == 'general_llm_cmdp':
                        dummy_vector = torch.zeros_like(vector_128d)
                        pred_type, e_t = collm.forward_and_get_uncertainty(dummy_vector, recent_10_str, top_k_str)
                    else:
                        pred_type, e_t = collm.forward_and_get_uncertainty(vector_128d, recent_10_str, top_k_str)
                    
                    if np.isnan(e_t): e_t = 0.5
                    if np.isnan(b_avail): b_avail = 100.0
                    if np.isnan(c_remain): c_remain = 0.5
                    
                    b_avail_norm = np.clip(b_avail / 250.0, 0.0, 1.0)
                    state = np.array([e_t, b_avail_norm, c_remain], dtype=np.float32)
                    
                    # 🟢 [실험 2] 의사결정(Decision) 비교군 로직
                    if args.mode == 'fixed_0': a_t = 0.0
                    elif args.mode == 'fixed_50': a_t = 0.5
                    elif args.mode == 'fixed_100': a_t = 1.0
                    elif args.mode == 'heuristic_cmdp':
                        a_t = np.clip(b_avail_norm * 0.8, 0.0, 1.0)
                    else:
                        a_t = agent.select_action(state, deterministic=False)
                    
                    next_state, reward, done, truncated, info = env.step(
                        action=[a_t], predicted_traffic_type=pred_type, b_avail=b_avail, 
                        e_t=e_t, pred_id=pred_id, real_req_id=real_req_id
                    )
                    
                    cost = info.get("cost", 0.0) 
                    cost_scaled = cost / 50.0
                    
                    # AI 학습 모델만 메모리 저장
                    if args.mode not in ['fixed_0', 'fixed_50', 'fixed_100', 'heuristic_cmdp']:
                        agent.memory.rewards.append(reward)
                        agent.memory.costs.append(cost_scaled) 
                        agent.memory.is_terminals.append(done)
                        
                        # 🟢 [타입 에러 완벽 픽스] Numpy를 Pytorch Tensor로 감싸서 GPU(device)에 할당
                        agent.memory.states.append(torch.tensor(state, dtype=torch.float32).to(device))
                        agent.memory.actions.append(torch.tensor([a_t], dtype=torch.float32).to(device))
                    
                    day_reward += reward
                    day_cost += cost               
                    day_cost_scaled += cost_scaled 
                    c_remain = next_state[2]
                    
                    ep_et_list.append(e_t)
                    ep_at_list.append(a_t)
                    
                    if step % 200 == 0:
                        print(f"[TRAIN] Ep {epoch}| Day {day}| Step {step:4d}| E_t: {e_t:.2f}| Action: {a_t:.2f}| Reward: {reward:.3f}")
                        
                    if done or truncated: break
                        
                if args.mode not in ['fixed_0', 'fixed_50', 'fixed_100', 'heuristic_cmdp']:
                    agent.train_step()
                    if args.mode != 'no_cmdp': 
                        agent.update_lagrangian(day_cost_scaled)
                
                ep_total_reward += day_reward
                ep_total_cost += day_cost 
                
            avg_ep_reward = ep_total_reward / TRAIN_DAYS
            avg_ep_cost = ep_total_cost / TRAIN_DAYS
            history_reward.append(avg_ep_reward)
            history_cost.append(avg_ep_cost)
            
            avg_et, avg_at = np.mean(ep_et_list), np.mean(ep_at_list)
            print(f"🎯 Epoch {epoch:2d}/{NUM_EPOCHS} 완료| 보상: {avg_ep_reward:.2f}| 위반: {avg_ep_cost:.2f}| 람다: {agent.lambda_val:.4f}\n")

            writer.add_scalar("1. Performance/Average_Reward", avg_ep_reward, epoch)
            writer.add_scalar("1. Performance/Average_Cost", avg_ep_cost, epoch)
            writer.add_scalar("2. Dynamics/Average_Action", avg_at, epoch)
            writer.add_scalar("2. Dynamics/Lagrangian_Lambda", agent.lambda_val, epoch)
            writer.add_scalar("3. Environment/Average_Uncertainty", avg_et, epoch)

        # --- TESTING PHASE (실전 트래픽 평가) ---
        print(f"\n📊 [PHASE 2] TESTING: 실전 트래픽 평가")
        test_rewards, test_costs = [], []
        
        for day in range(TRAIN_DAYS + 1, TRAIN_DAYS + TEST_DAYS + 1):
            ns3_state, _ = env.reset()
            c_remain = ns3_state[2] if len(ns3_state) > 2 else 1.0 
            day_reward, day_cost = 0.0, 0.0
            
            for step in range(MAX_STEPS_PER_DAY):
                absolute_step = TRAIN_DAYS * MAX_STEPS_PER_DAY + step
                ti = dataloader.get_real_traffic_at_t(absolute_step)
                
                b_avail_norm = np.clip(ti['b_avail'] / 250.0, 0.0, 1.0)
                state = np.array([0.5, b_avail_norm, c_remain], dtype=np.float32)
                
                a_t = agent.select_action(state, deterministic=True) if args.mode not in ['fixed_0', 'fixed_50', 'fixed_100'] else (0.5 if args.mode == 'fixed_50' else (1.0 if args.mode == 'fixed_100' else 0.0))
                
                _, reward, done, truncated, info = env.step(action=[a_t], predicted_traffic_type="VOD", b_avail=ti['b_avail'], e_t=0.5, pred_id=1, real_req_id=ti['item_id'])
                day_reward += reward
                day_cost += info.get("cost", 0.0)
                
                if done or truncated: break
            
            test_rewards.append(day_reward); test_costs.append(day_cost)
            print(f"🏁 [TEST] Day {day} | 보상: {day_reward:.2f} | 위반: {day_cost:.2f}")

        # --- [시각화 1] 이중 Y축 수렴 그래프 ---
        fig, ax1 = plt.subplots(figsize=(12, 5))
        color1 = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Average Reward', color=color1)
        ax1.plot(range(1, len(history_reward)+1), history_reward, label='Reward', color=color1, marker='o')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Average Cost', color=color2)
        ax2.plot(range(1, len(history_cost)+1), history_cost, label='Cost', color=color2, marker='x')
        ax2.axhline(y=0.0, color='r', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title(f'[{args.mode.upper()}] Training Curve')
        fig.tight_layout(); plt.grid(True)
        plt.savefig(f'train_convergence_{args.mode}.png')

        # --- [시각화 2] 에이전트 다이나믹스 ---
        # 🟢 [잠재적 에러 사전 차단] Tensor 리스트를 numpy 배열로 안전하게 변환
        if len(agent.memory.states) > 0:
            last_states = np.array([s.cpu().numpy() for s in agent.memory.states])
            last_actions = np.array([a.cpu().numpy() for a in agent.memory.actions])
            
            plt.figure(figsize=(12, 6))
            plt.plot(last_states[-MAX_STEPS_PER_DAY:, 0], label='Uncertainty (E_t)', color='orange')
            plt.plot(last_states[-MAX_STEPS_PER_DAY:, 1], label='Bandwidth (B_avail)', color='green')
            plt.plot(last_actions[-MAX_STEPS_PER_DAY:], label='Action (a_t)', color='blue', linewidth=2)
            plt.title('Agent Dynamics (Last Training Day)')
            plt.legend(); plt.grid(True)
            plt.savefig(f'decision_dynamics_{args.mode}.png')

    finally:
        env.close()
        writer.close()
        gc.collect()
        torch.cuda.empty_cache()
        print("📊 실험 종료 및 자원 정리 완료.")

if __name__ == "__main__":
    main()