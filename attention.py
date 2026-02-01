class MultiHeadAttention(nn.Module):

    def __init__(self,n_embed, head_num, block_size, dropout = 0.1):
        super().__init__()
        self.q  = nn.Linear(n_embed, n_embed)
        self.k = nn.Linear(n_embed, n_embed)
        self.v = nn.Linear(n_embed, n_embed)
        self.n_embed = n_embed
        self.head_num = head_num
        self.proj = nn.Linear(n_embed, n_embed)
        self.register_buffer('tril', torch.tril(torch.oness(block_size, block_size)))

    def forward(self,q,k,v,mask=None):
        B, T, C = q.shape
        self.head_size = self.n_embed // self.head_num
        Q = self.q(q)
        K = self.k(k)
        V = self.v(v)
        Q = Q.view(B, T, self.head_num, self.head_size).transpose(1,2) # B, head_num, T, head_szie
        K = K.view(B, T, self.head_num, self.head_size).transpose(1,2)
        V = V.view(B, T, self.head_num, self.head_size).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-1,-2))
        scores = scores / self.head_size **(0.5)
        if mask:
            scores = scores.masked_fill(self.tril[:T,:T] ==0, float('-inf'))
            

        softmax_scores = F.softmax(scores, dim=-1)
        attn_weights = torch.matmal(softmax_scores, V) / self.head_size **(-0.5)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # [B, head_num, T, head_size]

        # Concatenate heads: [B, T, head_num * head_size] = [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        out = self.proj(out)
        return out
