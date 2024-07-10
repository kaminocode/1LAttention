import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import random
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

block_size = config['block_size']
n_embd = config['n_embd']
n_head = config['n_head']
vocab_size = config['vocab_size']

def data_generator(batch_size, n_digits, PLUS_INDEX=10, EQUALS_INDEX=11, seed=1111):
    torch.manual_seed(seed)
    while True:
        batch = torch.zeros((batch_size, 3 * n_digits + 3), dtype=torch.int64)
        x = torch.randint(0, 10, (batch_size, n_digits))
        y = torch.randint(0, 10, (batch_size, n_digits))

        if random.randint(1, 5) == 1:
            indices_to_modify = torch.randperm(x.numel())[:int(0.20 * x.numel())]
            if random.randint(1, 2) == 1:
                x.view(-1)[indices_to_modify] = 9 - y.view(-1)[indices_to_modify]
            else:
                y.view(-1)[indices_to_modify] = 9 - x.view(-1)[indices_to_modify]

        batch[:, :n_digits] = x
        batch[:, n_digits] = PLUS_INDEX
        batch[:, 1 + n_digits:1 + n_digits * 2] = y
        batch[:, 1 + n_digits * 2] = EQUALS_INDEX

        carry = torch.zeros(batch_size, dtype=torch.int64)
        for i in range(n_digits):
            idx = -(i + 1)
            digit_sum = x[:, idx] + y[:, idx] + carry
            batch[:, idx] = digit_sum % 10
            carry = digit_sum // 10

        batch[:, -1 - n_digits] = carry
        yield batch

class AttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        # self.ffwd = FeedFoward(n_embd)

    def forward(self, x):
        x = x + self.sa(x)
        # x = x + self.ffwd(x)
        return x
    

class OneLayerAttentionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.attention = Block(n_embd, n_head)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.attention(x)

        logits = self.lm_head(x)

        return logits
