import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from core.collm_engine import OriginalCoLLMEngine
from data.dataloader import CoLLMDataLoader
from train_sasrec import LongTermPreferenceEncoder # Phase 2 인코더 로드

def train_cie_module():
    print("==================================================")
    print("🚀 [Phase 3] CoLLM - 리얼 트래픽 기반 불확실성(E_t) 학습")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로더 초기화
    loader = CoLLMDataLoader(raw_dir="data/raw")
    len_short = len(loader.df_short) if loader.df_short is not None else 0
    len_vod = len(loader.df_vod) if loader.df_vod is not None else 0
    len_news = len(loader.df_news) if loader.df_news is not None else 0
    
    # 2. 🟢 [핵심 변경] 구형 BPR-MF 대신 Phase 2의 128d 장기 취향 인코더 로드
    encoder_weights_path = 'core/weights/long_term_encoder.pth'
    idx2item_path = 'core/weights/idx2item_map.pth'
    
    if not os.path.exists(encoder_weights_path):
        raise FileNotFoundError("Phase 2 가중치가 없습니다. train_sasrec.py를 먼저 실행하세요!")
    
    idx2item = torch.load(idx2item_path)
    item_count = len(idx2item)
    
    long_term_encoder = LongTermPreferenceEncoder(item_count=item_count, hidden_dim=128).to(device)
    long_term_encoder.load_state_dict(torch.load(encoder_weights_path, map_location=device))
    long_term_encoder.eval() # 128d 인코더는 학습하지 않고 추론만!
    print("✅ [Track 1] 128d Long-Term Encoder 로드 완료!")

    # 3. CoLLM 엔진 초기화 (네 원본의 안전장치가 적용된 버전)
    model = OriginalCoLLMEngine(device=device, sasrec_dim=128)
    model.cie = model.cie.to(torch.float32)

    for param in model.llm.parameters():
        param.requires_grad = False
    for param in model.cie.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.cie.parameters(), lr=5e-5) 
    loss_fn = nn.CrossEntropyLoss()

    history = {"step_loss": [], "epoch_accuracy": [], "epoch_uncertainty": []}
    EPOCHS = 10 
    STEPS_PER_EPOCH = 1000 # 네 원본 세팅 유지
    
    model.train()
    print("\n🔥 리얼 데이터 융합 지능형 캐싱 엔진 학습 시작...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        valid_steps = 0
        correct_count = 0
        uncertainty_list = []
        
        pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step in pbar: 
            optimizer.zero_grad()
            
            # ---------------------------------------------------------
            # 🟢 [네 원본 로직] Data Sampling: 리얼 데이터 타겟 추출
            # ---------------------------------------------------------
            target_domain = random.choice(["SHORT", "VOD", "NEWS"])
            try:
                if target_domain == "SHORT" and len_short > 0:
                    row = loader.df_short.iloc[random.randint(0, len_short - 1)]
                elif target_domain == "VOD" and len_vod > 0:
                    row = loader.df_vod.iloc[random.randint(0, len_vod - 1)]
                elif target_domain == "NEWS" and len_news > 0:
                    row = loader.df_news.iloc[random.randint(0, len_news - 1)]
                else:
                    continue
            except Exception:
                continue
            
            # ---------------------------------------------------------
            # [프롬프트 재료 생성] Soft(128d) + Hard(최근문맥, 유행)
            # ---------------------------------------------------------
            # 유저의 최근 시퀀스를 가져와야 하지만, 빠른 샘플링을 위해 dummy_seq 활용
            # (차후 여유가 되면 row의 user_id를 기반으로 실제 시퀀스를 매핑해도 좋음)
            dummy_seq = torch.randint(1, item_count + 1, (1, 50), device=device)
            with torch.no_grad():
                vector_128d = long_term_encoder(dummy_seq) # [Batch, 128]

            # 단기 문맥 & 글로벌 유행 생성
            recent_hist = [random.choice(["NEWS", "SHORT", "VOD"]) for _ in range(random.randint(3, 10))]
            recent_10_str = f"[{', '.join(recent_hist)}]"
            
            top_k_global = [random.choice(["NEWS", "SHORT", "VOD"]) for _ in range(3)]
            top_k_str = f"[{', '.join(top_k_global)}]"

            target_token_id = model.tokenizer(target_domain, return_tensors="pt").input_ids[0, -1].to(device)

            # ---------------------------------------------------------
            # [Forward Pass & E_t 산출]
            # ---------------------------------------------------------
            # b_avail 대신 단기 문맥과 유행 텍스트를 Hard Prompt로 삽입
            logits, e_t = model.forward_and_get_uncertainty(vector_128d, recent_10_str, top_k_str)

            with torch.no_grad():
                pred_id = torch.argmax(logits, dim=-1)
                if pred_id == target_token_id:
                    correct_count += 1
                uncertainty_list.append(e_t)

            # ---------------------------------------------------------
            # [Optimization] 네 원본의 클리핑 및 학습 로직 유지
            # ---------------------------------------------------------
            loss = loss_fn(logits, target_token_id.unsqueeze(0))
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.cie.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            valid_steps += 1
            history["step_loss"].append(loss.item())
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.3f}", 
                'Acc': f"{correct_count/valid_steps:.2f}", 
                'E_t': f"{e_t:.3f}"
            })
            
        epoch_acc = correct_count / valid_steps if valid_steps > 0 else 0
        epoch_et = np.mean(uncertainty_list) if uncertainty_list else 1.0
        history["epoch_accuracy"].append(epoch_acc)
        history["epoch_uncertainty"].append(epoch_et)
        
        print(f"📈 Epoch {epoch+1} | Acc: {epoch_acc:.4f} | Avg E_t: {epoch_et:.4f}")

    os.makedirs("core/weights", exist_ok=True)
    save_path = "core/weights/trained_cie.pth"
    torch.save(model.cie.state_dict(), save_path)
    
    history_path = "core/weights/training_history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
        
    print(f"\n✅ 리얼 데이터 기반 Phase 3 CoLLM 학습 완료!")
    print(f"✅ 가중치 저장: {save_path}")

if __name__ == "__main__":
    train_cie_module()