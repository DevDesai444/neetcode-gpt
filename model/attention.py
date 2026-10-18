import torch
import torch.nn as nn
from torchtyping import TensorType

'''
Q -> (T, A) -> whats it looking for
K -> (T, A) -> what it have
V -> (T, A) -> another level of complexity, what it token willing to share, relavent ?
'''

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        # pass
        self.K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.V = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        # pass
        q,k,v = self.Q(embedded), self.K(embedded), self.V(embedded) # embedded -> (Batchs, no. Tokens, embedding_dim), q,k,v -> (Batchs, no. Tokens, attention_dim)
        temp = q @ torch.transpose(k,1,2) # (Batchs, no. Tokens, no. Tokens)
        scours = temp / (self.attention_dim ** 0.5)

        cMask = torch.tril(torch.ones(temp.shape[1], temp.shape[1]))
        scours = scours.masked_fill(cMask == 0, float('-inf'))

        scours = nn.functional.softmax(scours, dim = 2)
        attention = scours @ v

        return torch.round(attention, decimals = 4)