import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class CIE_Module(nn.Module):
    def __init__(self, sasrec_dim=128, llm_dim=4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(sasrec_dim, llm_dim // 2),
            nn.GELU(),
            nn.Linear(llm_dim // 2, llm_dim)
        )

    def forward(self, sasrec_vector):
        projected = self.projector(sasrec_vector)
        return projected.unsqueeze(1) 

class OriginalCoLLMEngine(nn.Module):
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", sasrec_dim=128, device="cuda"):
        super().__init__()
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        
        self.llm_dim = self.llm.config.hidden_size
        
        self.cie = CIE_Module(sasrec_dim=sasrec_dim, llm_dim=self.llm_dim).to(device)
        self.cie = self.cie.to(torch.float16) 
        
        # 🟢 [추가] 타겟 마스킹용 토큰 ID 미리 추출 (32000지선다 -> 3지선다)
        self.target_words = ["NEWS", "SHORT", "VOD"]
        self.target_ids = [self.tokenizer(word, add_special_tokens=False).input_ids[-1] for word in self.target_words]

    def build_text_prompt(self, recent_10_context, top_k_global):
        prompt = (
            f"Recent Context: {recent_10_context}. "
            f"Global Trending: {top_k_global}. "
            f"Predict next content type:"
        )
        return prompt

    def forward_and_get_uncertainty(self, vector_128d, recent_10_context, top_k_global):
        text_prompt = self.build_text_prompt(recent_10_context, top_k_global)
        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
        
        text_embeds = self.llm.get_input_embeddings()(inputs.input_ids)
        
        vector_128d = vector_128d.to(self.device, dtype=torch.float16) 
        collaborative_embeds = self.cie(vector_128d)
        collaborative_embeds = collaborative_embeds.to(text_embeds.dtype)
        
        inputs_embeds = torch.cat([collaborative_embeds, text_embeds], dim=1)
        
        outputs = self.llm(inputs_embeds=inputs_embeds)
        logits = outputs.logits[:, -1, :] 
        
        with torch.no_grad():
            # 🟢 [타겟 마스킹 적용] 딱 3개(NEWS, SHORT, VOD)의 로짓만 추출
            target_logits = logits[:, self.target_ids]
            
            # 3개 선택지 안에서만 100% 비중으로 확률(Softmax) 재계산
            probs = F.softmax(target_logits, dim=-1).to(torch.float32) 
            
            # 엔트로피 계산 (정규화 분모를 log(3)으로 변경하여 0.0~1.0 스케일 맞춤)
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=-1)
            max_entropy = torch.log(torch.tensor(probs.size(-1), dtype=torch.float32))
            et = (entropy / max_entropy).item()
            
            # 🟢 [버그 픽스] main.py가 받을 수 있게 가장 확률 높은 '카테고리 문자열'을 반환!
            pred_idx = torch.argmax(probs, dim=-1).item()
            pred_type = self.target_words[pred_idx]
            
        return pred_type, et