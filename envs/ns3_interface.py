import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ns3-ai 구버전 충돌 방지 핫픽스
np.float = float    
np.int = int        
np.bool = bool      

import ns3ai_gym_env

class EdgeCachingCMDPEnv(gym.Env):
    def __init__(self, target_name="edge_caching_sim"):
        super().__init__()
        
        self.action_space = spaces.Box(low=0.0, high=100000.0, shape=(3,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]), 
            high=np.array([1.0, 1000.0, 1.0]), 
            dtype=np.float32
        )

        self.ns3_env = gym.make(
            "ns3ai_gym_env/Ns3-v0", 
            targetName=target_name, 
            ns3Path="/home/jinho/workspace/ns-allinone-3.40/ns-3.40/"
        )
        self.traffic_weights = {"NEWS": 1, "SHORT": 10, "VOD": 1000}

    def step(self, action, predicted_traffic_type, b_avail, e_t, pred_id):
        a_t = action[0]
        
        req_bw = self.traffic_weights.get(predicted_traffic_type, 1)
        cost_a_t = req_bw * a_t
        violation = max(0.0, cost_a_t - b_avail) 

        ns3_obs, ns3_reward, done, truncated, info = self.ns3_env.step([a_t, b_avail, float(pred_id)])
        
        # 🟢 [핵심 버그 수정] info.get(...)의 가짜 0.1을 버리고, C++가 넘겨준 '진짜 보상'을 사용!
        reward = float(ns3_reward) 
        
        c_remain = ns3_obs[0] if len(ns3_obs) > 0 else 0.5 
        
        next_state = np.array([e_t, b_avail, c_remain], dtype=np.float32)
        return next_state, reward, done, truncated, {"cost": violation}

    def reset(self, seed=None, options=None):
        ns3_obs, info = self.ns3_env.reset()
        initial_state = np.array([0.0, 100.0, 1.0], dtype=np.float32)
        return initial_state, info