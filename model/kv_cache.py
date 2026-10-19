class KVCache:
    def __init__(self):
        self.cache_k = None
        self.cache_v = None

    def update(self, new_k, new_v):
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None


class CachedAttention(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.model_dim = model_dim

    def forward(self, x, kv_cache=None):
        q = self.q_proj(x)        # (batch, new_seq_len, model_dim)
        k = self.k_proj(x)        # (batch, new_seq_len, model_dim)
        v = self.v_proj(x)        # (batch, new_seq_len, model_dim)

        if kv_cache is None:
            kv_cache = KVCache()

        # how many tokens already in cache before this call
        prev_len = kv_cache.cache_k.shape[1] if kv_cache.cache_k is not None else 0

        # append new K, V to cache
        full_k, full_v = kv_cache.update(k, v)
        # full_k: (batch, total_seq_len, model_dim)

        # attention scores
        scores = (q @ full_k.transpose(1, 2)) / (self.model_dim ** 0.5)
        # (batch, new_seq_len, total_seq_len)

        # causal mask with offset
        new_len = q.shape[1]
        total_len = full_k.shape[1]
        mask = torch.zeros(new_len, total_len)
        for i in range(new_len):
            # token i (which is actually at position prev_len + i) can see
            # positions 0 through prev_len + i
            mask[i, prev_len + i + 1:] = float('-inf')
        scores = scores + mask

        # softmax
        weights = torch.softmax(scores, dim=-1)
        out = weights @ full_v

        return torch.round(out, decimals=4), kv_cache