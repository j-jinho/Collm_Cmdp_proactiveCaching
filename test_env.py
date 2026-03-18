import numpy as np
from core.cmdp_env import CoLLM_CMDP_Env

def main():
    print("🚀 CoLLM-CMDP 환경 연동 테스트를 시작합니다...")
    env = CoLLM_CMDP_Env(vocab_size=1000, device="cpu")

    # 🟢 gymnasium 규격에 맞게 반환값 2개 받기
    obs, info = env.reset()
    print(f"🌟 초기 State (S_0): {obs}")

    for step in range(1, 11):
        action = env.action_space.sample() 
        # 🟢 gymnasium 규격에 맞게 5개 받기
        next_obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n--- Step {step} ---")
        print(f"Action: {action}, State: {next_obs}, Reward: {reward}, Cost: {info.get('cost', 0.0)}")

        if terminated or truncated:
            break

    env.close()
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    main()