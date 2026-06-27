import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.module import multiTimeAttention 


class enc_mtan_rnn(nn.Module):
    def __init__(self, input_dim, n_ref_point, nhidden=16, embed_time=16, num_heads=1):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.nhidden = nhidden
        self.n_ref_point = n_ref_point

        self.register_buffer('query', torch.linspace(0, 1., self.n_ref_point))
        
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden))
        self.periodic = nn.Linear(1, embed_time-1)
        self.linear = nn.Linear(1, 1)
        
    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
    def forward(self, x, mask, time_steps):
        x = torch.cat((x, mask), dim=2)
        mask = torch.cat((mask, mask), dim=2)
        
        key = self.learn_time_embedding(time_steps)
        query = self.learn_time_embedding(self.query.unsqueeze(0))
        
        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out


class dec_mtan_rnn(nn.Module):
    def __init__(self, input_dim, n_ref_point, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.nhidden = nhidden
        self.n_ref_point = n_ref_point
        
        self.register_buffer('query', torch.linspace(0, 1., self.n_ref_point))
        
        self.att = multiTimeAttention(2*nhidden, 2*nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)    
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden))
        
        self.periodic = nn.Linear(1, embed_time-1)
        self.linear = nn.Linear(1, 1)
        
    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        
        query = self.learn_time_embedding(time_steps)
        key = self.learn_time_embedding(self.query.unsqueeze(0))
        
        out = self.att(query, key, out)
        out = self.z0_to_obs(out)
        return out        


class DeltaPredictor(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, embed_time=16, num_heads=1):
        super().__init__()
        
        self.encoder = enc_mtan_rnn(
            input_dim=input_dim, query=query, latent_dim=latent_dim,
            nhidden=nhidden, embed_time=embed_time, num_heads=num_heads
        )
        
        self.decoder = dec_mtan_rnn(
            input_dim=input_dim, query=query, latent_dim=latent_dim,
            nhidden=nhidden, embed_time=embed_time, num_heads=num_heads
        )

    def forward(self, x, mask, time):
        z = self.encoder(x, mask, time)
        preds = self.decoder(z, mask, time)
        
        return preds


if __name__ == '__main__':
    # 測試程式碼
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
    B, src_len, tgt_len, input_dim = 4, 10, 5, 3
    query = torch.linspace(0, 1, 8).to(device)
    
    # 初始化模型
    model = DeltaPredictor(input_dim=input_dim, query=query).to(device)
    
    # 模擬數據
    mock_x = torch.randn(B, src_len, 2 * input_dim).to(device)
    mock_src_time = torch.stack([torch.linspace(0, 0.7, src_len) for _ in range(B)]).to(device)
    mock_tgt_time = torch.stack([torch.linspace(0.7, 1.0, tgt_len) for _ in range(B)]).to(device)
    
    # 前向傳播
    predictions = model(mock_x, mock_src_time, mock_tgt_time)
    print("預測輸出形狀 (B, tgt_len, input_dim):", predictions.shape)