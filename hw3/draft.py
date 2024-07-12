import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3*cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
    def forward(self, x, visual_features):
        B, T, C = x.size()
        _, V, _ = visual_features.size()
        q = self.c_attn(x)[:, :, :self.n_embd]
        kv = self.c_attn(visual_features).view(B, V, 2, self.n_head, C // self.n_head)
        k, v = kv[:, :, 0], kv[:, :, 1]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))
    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))
class AdapterLayer(nn.Module):
    def __init__(self, input_size, intermediate_size):
        super().__init__()
        self.down_project = nn.Linear(input_size, intermediate_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(intermediate_size, input_size)
    def forward(self, x):
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x
class ModifiedBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.cross_attn = CrossAttention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))
    def forward(self, x, visual_features):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), visual_features)
        x = x + self.mlp(self.ln_3(x))
        return x
class BlockWithAdapter(nn.Module):
    def __init__(self, cfg, intermediate_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.adapter1 = AdapterLayer(cfg.n_embd, intermediate_size)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))
        self.adapter2 = AdapterLayer(cfg.n_embd, intermediate_size)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.adapter1(x)
        x = x + self.mlp(self.ln_2(x))
        x = x + self.adapter2(x)
        return x
class ModifiedDecoder(nn.Module):
    def __init__(self, cfg, intermediate_adapter_size, image_feature_size):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.Sequential(*[BlockWithAdapter(cfg, intermediate_adapter_size) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.image_feature_size = image_feature_size
        self.image_feature_transform_weight = nn.Parameter(torch.randn(cfg.n_embd, image_feature_size))
        if cfg.checkpoint is not None:
            state_dict = torch.load(cfg.checkpoint)
            transposed = ['.c_attn.weight', '.c_fc.weight', '.c_proj.weight']
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)
    def forward(self, text_input: Tensor, image_features: Tensor):
        text_embeddings = self.transformer.wte(text_input)
        transformed_image_features = nn.functional.linear(image_features, self.image_feature_transform_weight)
        transformed_image_features = transformed_image_features.unsqueeze(1).expand(-1, text_embeddings.size(1), -1)
        combined_embeddings = text_embeddings + transformed_image_features
        pos = torch.arange(text_input.size(1), dtype=torch.long, device=text_input.device).unsqueeze(0)
        x = combined_embeddings + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits