import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

from data.dataloader import CoLLMDataLoader

class LongTermPreferenceEncoder(nn.Module):
    def __init__(self, item_count, max_seq_len=50, hidden_dim=128):
        """
        [절대 원칙 2 & 4] 128d Soft Prompt용 인코더. OOM 방지를 위해 sparse=True 적용.
        """
        super().__init__()
        self.item_count = item_count
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # 🚨 Sparse 임베딩으로 1,000만 개 아이템 VRAM 초과 방지
        self.item_emb = nn.Embedding(item_count + 1, hidden_dim, padding_idx=0, sparse=True)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, input_seqs):
        # [Batch, Seq_len, 128]
        seq_emb = self.item_emb(input_seqs) 
        
        positions = torch.arange(input_seqs.size(1), device=input_seqs.device).unsqueeze(0).expand_as(input_seqs)
        seq_emb += self.pos_emb(positions)
        
        # Causal Mask (미래 정보 참조 방지)
        mask = nn.Transformer.generate_square_subsequent_mask(self.max_seq_len).to(input_seqs.device)
        
        encoded_seq = self.transformer(seq_emb, mask=mask)
        
        # 가장 마지막 시점의 128차원 벡터 반환 (장기 취향 요약본)
        long_term_vector = encoded_seq[:, -1, :] 
        return long_term_vector

def train_long_term_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    loader_engine = CoLLMDataLoader(raw_dir="data/raw")
    max_seq_len = 50
    # DataLoader 호출 (기존에 작성된 get_sasrec_loaders 사용)
    train_loader, item_count, idx2item = loader_engine.get_sasrec_loaders(batch_size=512, max_len=max_seq_len)
    
    model = LongTermPreferenceEncoder(item_count=item_count, max_seq_len=max_seq_len, hidden_dim=128).to(device)
    bce_criterion = nn.BCEWithLogitsLoss()
    
    # 🚨 [핵심] SparseAdam의 2.44GB 할당 버그를 피하기 위해 임베딩은 SGD 사용
    emb_params = list(model.item_emb.parameters())
    other_params = [p for n, p in model.named_parameters() if 'item_emb' not in n]
    
    optimizer_sparse = optim.SGD(emb_params, lr=0.1) 
    optimizer_dense = optim.Adam(other_params, lr=0.001)

    epochs = 10
    print(f"🚀 [Phase 2] Starting 128d Long-Term Encoder Training... (Total items: {item_count:,})")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            _, input_seqs, targets = [x.to(device) for x in batch]
            
            valid_mask = (targets != 0)
            if valid_mask.sum() == 0:
                continue
            
            optimizer_sparse.zero_grad()
            optimizer_dense.zero_grad()
            
            # 1. 128d 장기 취향 벡터 추출
            long_term_vector = model(input_seqs) 
            
            # 2. Negative Sampling을 통한 Loss 계산 (MatMul OOM 회피)
            pos_emb = model.item_emb(targets)
            pos_logits = (long_term_vector * pos_emb).sum(dim=-1)
            
            neg_items = torch.randint(1, model.item_count + 1, targets.shape, device=device)
            neg_emb = model.item_emb(neg_items)
            neg_logits = (long_term_vector * neg_emb).sum(dim=-1)
            
            pos_loss = bce_criterion(pos_logits[valid_mask], torch.ones_like(pos_logits[valid_mask]))
            neg_loss = bce_criterion(neg_logits[valid_mask], torch.zeros_like(neg_logits[valid_mask]))
            
            loss = pos_loss + neg_loss
            loss.backward()
            
            optimizer_sparse.step()
            optimizer_dense.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"✅ Epoch {epoch+1} Completed | Average Loss: {total_loss/len(train_loader):.4f}")

    # 가중치 저장 (Phase 3에서 CoLLM의 Soft Prompt 생성 시 활용)
    os.makedirs('core/weights', exist_ok=True)
    torch.save(model.state_dict(), 'core/weights/long_term_encoder.pth')
    torch.save(idx2item, 'core/weights/idx2item_map.pth')
    print("💾 128d Encoder Model & ID Map saved to core/weights/long_term_encoder.pth")

if __name__ == "__main__":
    train_long_term_encoder()