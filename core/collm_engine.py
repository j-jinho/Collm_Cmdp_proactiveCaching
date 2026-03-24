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
        
        # =========================================================
        # 🟢 [최종 픽스] device_map 옵션을 아예 삭제해서 웜업(Warmup) 버그 차단!
        # CPU에 16-bit로 먼저 가볍게 올린 뒤, GPU로 한 번에 넘깁니다. 
        # (2.2GB라 WSL 통과 가능)
        # =========================================================
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        ).to(device)
        
        self.llm_dim = self.llm.config.hidden_size
        
        self.cie = CIE_Module(sasrec_dim=sasrec_dim, llm_dim=self.llm_dim).to(device)
        self.cie = self.cie.to(torch.float16) 
        
        # 타겟 마스킹용 토큰 ID 미리 추출 (32000지선다 -> 3지선다)
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
            target_logits = logits[:, self.target_ids]
            
            # 🟢 Temperature Scaling (0.1) 적용
            temperature = 0.1
            probs = F.softmax(target_logits / temperature, dim=-1).to(torch.float32) 
            
            # 엔트로피 계산
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=-1)
            max_entropy = torch.log(torch.tensor(probs.size(-1), dtype=torch.float32))
            et = (entropy / max_entropy).item()
            
            pred_idx = torch.argmax(probs, dim=-1).item()
            pred_type = self.target_words[pred_idx]
            
        return pred_type, et