import gymnasium as gym 
from gymnasium import spaces
import numpy as np

np.float = float    
np.int = int        
np.bool = bool      

import ns3ai_gym_env 

class CoLLM_CMDP_Env(gym.Env):
    def __init__(self, target_name="edge_caching_sim", ns3_path="/home/jinho/workspace/ns-allinone-3.40/ns-3.40/"):
        super(CoLLM_CMDP_Env, self).__init__()
        
        # 🟢 [절대 원칙 3] Action: 선제적 캐싱 비율 a_t (0.0 ~ 1.0 연속값)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 🟢 [절대 원칙 3] State: [E_t(불확실성), B_avail(대역폭), C_remain(캐시잔량)]
        low_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        high_state = np.array([10.0, 10000.0, 1.0], dtype=np.float32) # E_t max를 여유있게 10.0으로
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)

        self.env = gym.make(
            "ns3ai_gym_env/Ns3-v0", 
            targetName=target_name, 
            ns3Path=ns3_path
        )
        
        # 🟢 [절대 원칙 1] 이질적 트래픽 용량 (Cost 계산용)
        self.traffic_weights = {"NEWS": 1.0, "SHORT": 10.0, "VOD": 1000.0}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # 초기 상태: E_t=0.0, B_avail=100.0, C_remain=1.0 가정
        state = np.array([0.0, 100.0, 1.0], dtype=np.float32)
        return state, info

    def step(self, action, current_e_t, current_domain):
        """
        action: PPO가 결정한 캐싱 비율 a_t [0, 1]
        current_e_t: CoLLM이 방금 뱉어낸 불확실성 E_t
        current_domain: 현재 트래픽 도메인 ("NEWS", "SHORT", "VOD")
        """
        a_t = np.clip(action[0], 0.0, 1.0)
        
        # ns-3 환경에 액션(비율) 전달 (구현된 C++ 규격에 맞게 전송 형태 조율 필요)
        # 만약 C++ 측이 a_t 하나만 받는다면 [a_t]를 넘김
        obs, reward, terminated, truncated, info = self.env.step([float(a_t)])
        
        # 🟢 [절대 원칙 1 & 3] Cost 계산: 트래픽 용량 * 캐싱 비율
        req_bw = self.traffic_weights.get(current_domain, 1.0)
        cost_a_t = req_bw * a_t
        
        # 남은 상태값 업데이트 (ns-3가 주는 obs 구조에 맞춰 수정 필요)
        c_remain = obs[0] if len(obs) > 0 else 0.5 
        b_avail = obs[1] if len(obs) > 1 else 100.0
        
        # 백홀 대역폭 초과 시 Violation Cost 발생
        violation = max(0.0, cost_a_t - b_avail)
        info['cost'] = violation
        
        next_state = np.array([current_e_t, b_avail, c_remain], dtype=np.float32)
        return next_state, reward, terminated, truncated, info

    def close(self):
        self.env.close()