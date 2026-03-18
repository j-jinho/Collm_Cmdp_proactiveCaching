import torch
import pickle
import os
from core.collm_engine import OriginalCoLLMEngine

def run_sanity_check():
    print("==================================================")
    print("🔬 [검증] 학습된 CoLLM 뇌세포(CIE) 추론 테스트")
    print("==================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 모델 엔진 및 가중치 로드
    print("⏳ 엔진 및 가중치 불러오는 중... (TinyLlama & trained_cie.pth)")
    model = OriginalCoLLMEngine(device=device)
    
    weights_path = "core/weights/trained_cie.pth"
    if not os.path.exists(weights_path):
        print("❌ 에러: trained_cie.pth 파일이 없습니다. 학습이 완료되었는지 확인하세요.")
        return
        
    # 학습된 프로젝터(CIE) 가중치 덮어쓰기
    model.cie.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.cie.float()
    model.eval() # 평가 모드 전환 (Dropout 등 비활성화)

    # 2. BPR 유저 임베딩 로드
    embed_path = "core/weights/user_embeddings.pkl"
    with open(embed_path, "rb") as f:
        user_embeddings = pickle.load(f)
    
    # 테스트용으로 0번 가상 유저의 128차원 벡터 추출
    test_mf_vector = torch.tensor(user_embeddings[0], dtype=torch.float32).unsqueeze(0).to(device)

    print("✅ 준비 완료! 테스트 시나리오를 시작합니다.\n")

    # ---------------------------------------------------------
    # 🧪 시나리오 1: 아주 확실하고 쾌적한 상황
    # ---------------------------------------------------------
    hist_1 = "[VOD, VOD, VOD]"
    bw_1 = 150.0 # 150 Mbps (매우 쾌적함)
    
    print(f"▶️ [시나리오 1] 유저0 | 히스토리: {hist_1} | 대역폭: {bw_1} Mbps")
    pred_1, et_1 = model.forward_and_get_uncertainty(test_mf_vector, hist_1, bw_1)
    print(f"   ㄴ 🤖 예측 결과: {pred_1}")
    print(f"   ㄴ 📊 불확실성(E_t): {et_1:.4f} (0에 가까울수록 확신함)\n")

    # ---------------------------------------------------------
    # 🧪 시나리오 2: 혼란스럽고 열악한 상황
    # ---------------------------------------------------------
    hist_2 = "[NEWS, SHORT, VOD]"
    bw_2 = 2.5 # 2.5 Mbps (매우 열악함)
    
    print(f"▶️ [시나리오 2] 유저0 | 히스토리: {hist_2} | 대역폭: {bw_2} Mbps")
    pred_2, et_2 = model.forward_and_get_uncertainty(test_mf_vector, hist_2, bw_2)
    print(f"   ㄴ 🤖 예측 결과: {pred_2}")
    print(f"   ㄴ 📊 불확실성(E_t): {et_2:.4f} (1에 가까울수록 혼란스러움)\n")
    
    print("==================================================")
    print("💡 분석: 시나리오 1의 E_t가 시나리오 2보다 낮게 나왔다면,")
    print("   모델이 상황의 '불확실성'을 제대로 이해하고 있다는 증거입니다!")
    print("==================================================")

if __name__ == "__main__":
    run_sanity_check()