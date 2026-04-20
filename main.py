# =========================================================
# 🔴 [CRITICAL] ns3-ai가 인자를 가로채기 전에 최상단에서 차단
# C++(ns-3)는 파이썬의 --mode, --epochs 같은 인자를 인식하지 못해 터집니다.
# 라이브러리 로드 전 sys.argv를 비워주는 것이 유일한 해결책입니다.
# =========================================================
import sys
_original_args = sys.argv[:]  
sys.argv = [sys.argv[0]]      

import argparse 
import numpy as np
import torch
import os
import gc
import time
import random
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from data.dataloader import CoLLMDataLoader
from core.collm_engine import OriginalCoLLMEngine
from core.cmdp_agent import PPOLagrangianAgent
from envs.ns3_interface import EdgeCachingCMDPEnv 
from train_sasrec import LongTermPreferenceEncoder 

from core.baselines import LSTMPredictor, TCNPredictor, TransformerPredictor, PopularityPredictor, get_prediction_and_uncertainty

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
    
    parser.add_argument('--mode', type=str, default='proposed', 
                        choices=[
                            'proposed',           
                            'lstm_cmdp',          
                            'tcn_cmdp',           
                            'transformer_cmdp',   
                            'general_llm_cmdp',   
                            'popularity_cmdp',    
                            'no_cmdp',            
                            'heuristic_cmdp',     
                            'fixed_0',            
                            'fixed_50',           
                            'fixed_100'           
                        ],
                        help='실험할 알고리즘 모드를 선택하세요.')
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args(_original_args[1:])

    print(f"🚀 [{args.mode.upper()}] 모드로 [리얼 데이터 동기화 완료] 엣지 캐싱 본 실험을 시작합니다... (Epochs: {args.epochs})")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    tb_log_dir = os.path.join(current_dir, "runs", f"Experiment_{args.mode}")
    os.makedirs(tb_log_dir, exist_ok=True) 
    
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"📊 TensorBoard 로깅 시작! 터미널에 'tensorboard --logdir {tb_log_dir}'을 입력하세요.")

    dataloader = CoLLMDataLoader(raw_dir="data/raw")

    idx2item_path = 'core/weights/idx2item_map.pth'
    idx2item = torch.load(idx2item_path)
    item_count = len(idx2item)
    item2idx = {item: idx for idx, item in idx2item.items()}

    collm = None
    encoder = None
    baseline_model = None

    if args.mode in ['proposed', 'general_llm_cmdp', 'no_cmdp', 'heuristic_cmdp', 'fixed_0', 'fixed_50', 'fixed_100']:
        print("🧠 [1/2] CoLLM (LLM) 엔진을 VRAM(GPU)에 적재합니다...")
        collm = OriginalCoLLMEngine(device=device, sasrec_dim=128)
        cie_weight_path = "core/weights/trained_cie.pth"
        if os.path.exists(cie_weight_path):
            collm.cie.load_state_dict(torch.load(cie_weight_path, map_location=device))
        collm.eval() 
        
        # 🟢 [메모리 최적화] 무거운 LLM을 올린 직후 VRAM 파편화 강제 청소 (WSL 튕김 방지)
        torch.cuda.empty_cache()
        gc.collect()

        # 🟢 [메모리 픽스] SASRec은 절대 GPU로 보내지 않고 CPU에 둡니다. (OOM 완벽 차단)
        if args.mode != 'general_llm_cmdp':
            print("🧠 [2/2] SASRec 인코더(128d)를 CPU RAM에 적재합니다...")
            encoder_path = 'core/weights/long_term_encoder.pth'
            encoder = LongTermPreferenceEncoder(item_count=item_count, hidden_dim=128)
            encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
            encoder.eval()
        else:
            print("🧠 [2/2] General LLM 모드: 순수 LLM 성능 평가를 위해 SASRec을 비활성화합니다.")

    elif args.mode == 'lstm_cmdp':
        print("🧠 LSTM 베이스라인 모델을 적재합니다 (초고속)...")
        baseline_model = LSTMPredictor(num_items=item_count).to(device) 
        if os.path.exists('core/weights/lstm_best.pth'):
            baseline_model.load_state_dict(torch.load('core/weights/lstm_best.pth', map_location=device))
        baseline_model.eval()

    elif args.mode == 'tcn_cmdp':
        print("🧠 TCN 베이스라인 모델을 적재합니다 (초고속)...")
        baseline_model = TCNPredictor(num_items=item_count).to(device) 
        if os.path.exists('core/weights/tcn_best.pth'):
            baseline_model.load_state_dict(torch.load('core/weights/tcn_best.pth', map_location=device))
        baseline_model.eval()

    elif args.mode == 'transformer_cmdp':
        print("🧠 Transformer 베이스라인 모델을 적재합니다 (초고속)...")
        baseline_model = TransformerPredictor(num_items=item_count).to(device) 
        if os.path.exists('core/weights/transformer_best.pth'):
            baseline_model.load_state_dict(torch.load('core/weights/transformer_best.pth', map_location=device))
        baseline_model.eval()

    elif args.mode == 'popularity_cmdp':
        print("📊 인기도 기반 통계 모델을 준비합니다 (초고속)...")
        baseline_model = PopularityPredictor()
        
    agent = PPOLagrangianAgent(
        state_dim=3, action_dim=1,
        lr_actor=3e-4, lr_critic=1e-3, lr_lambda=5e-3, gamma=0.99
    )
    
    print(f"🌐 C++ (ns-3) 통신 다리를 개통합니다...")
    env = EdgeCachingCMDPEnv(target_name="edge_caching_sim") 

    # 🟢 30분 단위 제어 및 4주 학습
    MAX_STEPS_PER_DAY = 48 
    TRAIN_DAYS = 28
    TEST_DAYS = 7
    
    try:
        NUM_EPOCHS = args.epochs if args.mode not in ['fixed_0', 'fixed_50', 'fixed_100', 'heuristic_cmdp'] else 1 
        
        history_reward, history_cost = [], []

        for epoch in range(1, NUM_EPOCHS + 1):
            ep_total_reward, ep_total_cost = 0.0, 0.0
            ep_et_list, ep_at_list, ep_b_avail_list = [], [], []
            
            # 🟢 [추가됨: 4대 핵심 지표 누적 변수]
            total_req, total_hits = 0, 0
            total_bytes, hit_bytes = 0.0, 0.0
            latencies = []
            ep_actual_cost_total = 0.0
            
            for day in range(1, TRAIN_DAYS + 1):
                # 🟢 [충돌 픽스] Linux 공유 메모리(SHM) 찌꺼기 강제 청소
                os.system("rm -f /dev/shm/ns3* > /dev/null 2>&1")
                time.sleep(0.2)
                
                ns3_state, info = env.reset()
                c_remain = ns3_state[2] if len(ns3_state) > 2 else 1.0 
                day_reward, day_cost, day_cost_scaled = 0.0, 0.0, 0.0
                
                for step in range(MAX_STEPS_PER_DAY):
                    absolute_step = ((day - 1) * MAX_STEPS_PER_DAY + step) * 30 
                    
                    traffic_info = dataloader.get_real_traffic_at_t(absolute_step)
                    b_avail = traffic_info['b_avail']
                    pred_domain = traffic_info['domain']
                    real_req_id = traffic_info['item_id'] 
                    user_id = traffic_info['user_id']
                    
                    real_seq_tensor = dataloader.get_user_seq_tensor(user_id, real_req_id, item2idx, max_len=50)
                    recent_hist = dataloader.get_dynamic_recent_domains(user_id, pred_domain)
                    recent_10_str = f"[{', '.join(recent_hist)}]"
                    top_k_str = f"[{pred_domain}, SHORT]"
                    
                    with torch.no_grad():
                        if args.mode in ['proposed', 'no_cmdp', 'heuristic_cmdp', 'fixed_0', 'fixed_50', 'fixed_100']:
                            real_seq_tensor_cpu = real_seq_tensor.cpu()
                            vector_128d_batch_cpu = encoder(real_seq_tensor_cpu) 
                            all_item_embeddings_cpu = encoder.item_emb.weight 
                            scores_cpu = torch.matmul(vector_128d_batch_cpu, all_item_embeddings_cpu.T) 
                            
                            scores_cpu[0, 0] = -float('inf') 
                            pred_idx = torch.argmax(scores_cpu, dim=-1).item()
                            pred_id = int(idx2item.get(pred_idx, pred_idx))
                            
                            vector_128d_batch_gpu = vector_128d_batch_cpu.to(device)
                            pred_type, e_t = collm.forward_and_get_uncertainty(vector_128d_batch_gpu, recent_10_str, top_k_str)
                            
                        elif args.mode == 'general_llm_cmdp':
                            dummy_vector = torch.zeros((1, 128), device=device)
                            pred_id = int(idx2item.get(1, 1))
                            pred_type, e_t = collm.forward_and_get_uncertainty(dummy_vector, recent_10_str, top_k_str)
                            
                        elif args.mode == 'popularity_cmdp':
                            baseline_model.update(real_req_id)
                            pred_id, e_t = baseline_model.predict_and_get_uncertainty()
                            pred_type = recent_10_str.strip('[]').split(',')[-1].strip() if recent_10_str != "[]" else "VOD"
                            
                        elif args.mode in ['lstm_cmdp', 'tcn_cmdp', 'transformer_cmdp']:
                            seq_input = real_seq_tensor.to(device)
                            if seq_input.dim() == 1:
                                seq_input = seq_input.unsqueeze(0)
                            
                            logits = baseline_model(seq_input)
                            pred_idx, e_t = get_prediction_and_uncertainty(logits)
                            pred_id = int(idx2item.get(pred_idx, pred_idx))
                            pred_type = recent_10_str.strip('[]').split(',')[-1].strip() if recent_10_str != "[]" else "VOD"
                    
                    if np.isnan(e_t): e_t = 0.5
                    if np.isnan(b_avail): b_avail = 100.0
                    if np.isnan(c_remain): c_remain = 0.5
                    
                    b_avail_norm = np.clip(b_avail / 250.0, 0.0, 1.0)
                    state = np.array([e_t, b_avail_norm, c_remain], dtype=np.float32)
                    
                    if args.mode == 'fixed_0': a_t = 0.0
                    elif args.mode == 'fixed_50': a_t = 0.5
                    elif args.mode == 'fixed_100': a_t = 0.99 
                    elif args.mode == 'heuristic_cmdp': 
                        a_t = np.clip(b_avail_norm * 0.8, 0.0, 0.99)
                    else:
                        a_t = agent.select_action(state, deterministic=False)
                        
                    a_t = float(np.clip(a_t, 0.0, 0.99))
                    
                    next_state, reward, done, truncated, info = env.step(
                        action=[a_t], predicted_traffic_type=pred_type, b_avail=b_avail, 
                        e_t=e_t, pred_id=pred_id, real_req_id=real_req_id
                    )
                    
                    cost = info.get("cost", 0.0) 
                    cost_scaled = cost / 50.0
                    
                    # 🟢 [추가됨: 4대 지표 산출을 위한 수학적 역산 로직]
                    # RandomState를 사용해 다른 np.random 호출과 간섭 없이 일관된 용량(MB) 부여
                    rng = np.random.RandomState(real_req_id)
                    content_size_mb = rng.exponential(50) + 10
                    
                    is_hit = (reward > 0)
                    lat = 5.0 if is_hit else 55.0
                    saved_b = content_size_mb if is_hit else 0.0
                    
                    total_req += 1
                    total_hits += (1 if is_hit else 0)
                    total_bytes += content_size_mb
                    hit_bytes += saved_b
                    latencies.append(lat)
                    ep_actual_cost_total += cost
                    # -----------------------------------------------

                    if args.mode not in ['fixed_0', 'fixed_50', 'fixed_100', 'heuristic_cmdp']:
                        agent.memory.rewards.append(reward)
                        agent.memory.costs.append(cost_scaled) 
                        agent.memory.is_terminals.append(done)
                    
                    day_reward += reward
                    day_cost += cost               
                    day_cost_scaled += cost_scaled 
                    c_remain = next_state[2]
                    
                    ep_et_list.append(e_t)
                    ep_at_list.append(a_t)
                    ep_b_avail_list.append(b_avail_norm)
                    
                    if step % 24 == 0: 
                        print(f"[TRAIN] Ep {epoch}| Day {day}| Step {step:4d}| E_t: {e_t:.2f}| Action: {a_t:.2f}| Reward: {reward:.3f}")
                        
                    if done or truncated: break
                        
                if args.mode not in ['fixed_0', 'fixed_50', 'fixed_100', 'heuristic_cmdp']:
                    agent.train_step()
                    if args.mode != 'no_cmdp': 
                        agent.update_lagrangian(day_cost_scaled)
                
                ep_total_reward += day_reward
                ep_total_cost += day_cost 
                
            # 🟢 [스케일 픽스] 누적 보상을 Hit Rate 백분율(%)로 정확히 변환!
            avg_ep_reward_raw = ep_total_reward / TRAIN_DAYS
            avg_ep_hit_rate_percent = (avg_ep_reward_raw / MAX_STEPS_PER_DAY) * 100.0
            
            avg_ep_cost = ep_total_cost / TRAIN_DAYS
            history_reward.append(avg_ep_hit_rate_percent)
            history_cost.append(avg_ep_cost)
            
            # 🟢 [추가됨: 논문용 고급 4대 지표 통합 계산]
            ep_chr = (total_hits / total_req) * 100.0 if total_req > 0 else 0
            ep_bhr = (hit_bytes / total_bytes) * 100.0 if total_bytes > 0 else 0
            ep_lat = np.mean(latencies) if latencies else 55.0
            ep_avg_daily_cost = ep_actual_cost_total / TRAIN_DAYS

            avg_et, avg_at = np.mean(ep_et_list), np.mean(ep_at_list)
            
            print(f"🎯 Epoch {epoch:2d}/{NUM_EPOCHS} 완료 | CHR: {ep_chr:.1f}% | BHR: {ep_bhr:.1f}% | Lat: {ep_lat:.1f}ms | Cost: {ep_avg_daily_cost:.1f} | 람다: {agent.lambda_val:.4f}\n")

            # 기존 텐서보드 지표 기록
            writer.add_scalar("1. Performance/Average_Hit_Rate_Percent", avg_ep_hit_rate_percent, epoch)
            writer.add_scalar("1. Performance/Average_Cost", avg_ep_cost, epoch)
            writer.add_scalar("2. Dynamics/Average_Action", avg_at, epoch)
            writer.add_scalar("2. Dynamics/Lagrangian_Lambda", agent.lambda_val, epoch)
            writer.add_scalar("3. Environment/Average_Uncertainty", avg_et, epoch)
            
            # 🟢 [추가됨: plot_results.py 연동을 위한 4대 핵심 지표 기록]
            writer.add_scalar("Metrics/1_CHR_Percent", ep_chr, epoch)
            writer.add_scalar("Metrics/2_BHR_Percent", ep_bhr, epoch)
            writer.add_scalar("Metrics/3_Average_Latency_ms", ep_lat, epoch)
            writer.add_scalar("Metrics/4_Average_Daily_Cost", ep_avg_daily_cost, epoch)

        # --- TESTING PHASE ---
        print(f"\n📊 [PHASE 2] TESTING: 실전 트래픽 평가")
        test_rewards, test_costs = [], []
        
        for day in range(TRAIN_DAYS + 1, TRAIN_DAYS + TEST_DAYS + 1):
            os.system("rm -f /dev/shm/ns3* > /dev/null 2>&1")
            time.sleep(0.2)
            
            ns3_state, _ = env.reset()
            c_remain = ns3_state[2] if len(ns3_state) > 2 else 1.0 
            day_reward, day_cost = 0.0, 0.0
            
            # 테스트 지표 변수
            t_req, t_hits, t_total_bytes, t_hit_bytes = 0, 0, 0.0, 0.0
            t_latencies = []
            
            for step in range(MAX_STEPS_PER_DAY):
                absolute_step = ((day - 1) * MAX_STEPS_PER_DAY + step) * 30 
                ti = dataloader.get_real_traffic_at_t(absolute_step)
                
                b_avail_norm = np.clip(ti['b_avail'] / 250.0, 0.0, 1.0)
                state = np.array([0.5, b_avail_norm, c_remain], dtype=np.float32)
                
                a_t = agent.select_action(state, deterministic=True) if args.mode not in ['fixed_0', 'fixed_50', 'fixed_100'] else (0.5 if args.mode == 'fixed_50' else (0.99 if args.mode == 'fixed_100' else 0.0))
                a_t = float(np.clip(a_t, 0.0, 0.99)) 
                
                _, reward, done, truncated, info = env.step(action=[a_t], predicted_traffic_type="VOD", b_avail=ti['b_avail'], e_t=0.5, pred_id=int(idx2item.get(1, 1)), real_req_id=ti['item_id'])
                
                day_reward += reward
                cost_val = info.get("cost", 0.0)
                day_cost += cost_val
                
                # 테스트 지표 역산
                rng = np.random.RandomState(ti['item_id'])
                content_size_mb = rng.exponential(50) + 10
                is_hit = (reward > 0)
                
                t_req += 1
                t_hits += (1 if is_hit else 0)
                t_total_bytes += content_size_mb
                t_hit_bytes += (content_size_mb if is_hit else 0.0)
                t_latencies.append(5.0 if is_hit else 55.0)
                
                if done or truncated: break
            
            day_hit_rate_percent = (day_reward / MAX_STEPS_PER_DAY) * 100.0
            test_rewards.append(day_hit_rate_percent); test_costs.append(day_cost)
            
            day_bhr = (t_hit_bytes / t_total_bytes) * 100.0 if t_total_bytes > 0 else 0
            day_lat = np.mean(t_latencies) if t_latencies else 55.0
            
            print(f"🏁 [TEST] Day {day} | CHR: {day_hit_rate_percent:.1f}% | BHR: {day_bhr:.1f}% | Lat: {day_lat:.1f}ms | 위반: {day_cost:.1f}")

        fig, ax1 = plt.subplots(figsize=(12, 5))
        color1 = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Average Hit Rate (%)', color=color1)
        ax1.plot(range(1, len(history_reward)+1), history_reward, label='Hit Rate (%)', color=color1, marker='o')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 100) # 그래프 Y축을 0~100으로 고정!

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Average Cost', color=color2)
        ax2.plot(range(1, len(history_cost)+1), history_cost, label='Cost', color=color2, marker='x')
        ax2.axhline(y=0.0, color='r', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title(f'[{args.mode.upper()}] Training Curve')
        fig.tight_layout(); plt.grid(True)
        plt.savefig(f'train_convergence_{args.mode}.png')

        if len(ep_et_list) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(ep_et_list[-MAX_STEPS_PER_DAY:], label='Uncertainty (E_t)', color='orange')
            plt.plot(ep_b_avail_list[-MAX_STEPS_PER_DAY:], label='Bandwidth (B_avail)', color='green')
            plt.plot(ep_at_list[-MAX_STEPS_PER_DAY:], label='Action (a_t)', color='blue', linewidth=2)
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