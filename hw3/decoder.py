import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import loralib as lora 


class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12 # multi-head attention
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        if checkpoint is not None:
            print('Loading checkpoint', checkpoint)

class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, cfg.n_embd)
        # self.value = lora.Linear(cfg.n_embd, cfg.n_embd, r=8)
        # self.query = lora.Linear(cfg.n_embd, cfg.n_embd, r=8)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.query = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.out = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
    
    def forward(self, text, image):
        B, T, C = text.size()
        _, V, _ = image.size() # I = head_dim * self.n_head, 768 = 64 * 12
        # print('image size', image.size())
        head_dim = C // self.n_head

        # Reshape and compute query, key, value
        k = self.key(image).view(B, V, self.n_head,  head_dim).transpose(1, 2)  
        v = self.value(image).view(B, V, self.n_head, head_dim).transpose(1, 2)  
        q = self.query(text).view(B, T, self.n_head, head_dim).transpose(1, 2)
    
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).contiguous().view(B, T, C)

        return self.out(y)

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.bn_attn = nn.BatchNorm1d(3 * cfg.n_embd)
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
        y = self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))
        return y 
    
class Attention_4_lora(nn.Module):
    def __init__(self, cfg, rank=64):
        super().__init__()
        self.rank = rank
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.bn_attn = nn.BatchNorm1d(3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

        # LoRA matrices for c_attn
        self.lora_attn_A = nn.Parameter(torch.randn(cfg.n_embd, rank))
        self.lora_attn_B = nn.Parameter(torch.randn(rank, 3 * cfg.n_embd))

        # LoRA matrices for c_proj
        self.lora_proj_A = nn.Parameter(torch.randn(cfg.n_embd, rank))
        self.lora_proj_B = nn.Parameter(torch.randn(rank, cfg.n_embd))

    def lora_linear(self, x, linear, A, B):
        # Apply LoRA adaptation
        W_lora = torch.matmul(A, B).view(linear.weight.size())
        W = linear.weight + W_lora
        return F.linear(x, W, linear.bias)

    def forward(self, x):
        B, T, C = x.size()

        # Apply LoRA to c_attn
        x_attn = self.lora_linear(x, self.c_attn, self.lora_attn_A, self.lora_attn_B)
        x_attn = self.bn_attn(x_attn.transpose(1, 2)).transpose(1, 2)

        q, k, v = x_attn.split(self.n_embd, dim=2)
        # ... rest of the forward method remains the same ...
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))
        # Apply LoRA to c_proj
        y = self.lora_linear(y, self.c_proj, self.lora_proj_A, self.lora_proj_B)
        return y


class Block(nn.Module):
    def __init__(self, cfg, add_adapter=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        # self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.attn_lora = Attention_4_lora(cfg) # LoRA attention
        self.cross_attn = CrossAttention(cfg) # cross attention
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))
        self.add_adapter = add_adapter
        if add_adapter:
            self.adapter1 = Adapter(cfg)

    def forward(self, text, image):
        # text = text + self.attn_lora(self.ln_1(text))
        text = text + self.attn(self.ln_1(text))
        text = text + self.cross_attn(text, image) # Apply cross attention ??layer norm to text??
        text = text + self.mlp(self.ln_2(text))
        if self.add_adapter:
            text = text + self.adapter1(text) # Apply adapter
        return text

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h =  nn.Sequential(
                *[Block(cfg) for _ in range(cfg.n_layer - 2)], 
                *[Block(cfg, add_adapter=True) for _ in range(2)]  
            ),
            # h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # load checkpoint if available
        if self.cfg.checkpoint is not None:
            # print('Loading checkpoint', self.cfg.checkpoint)
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)
    
    def freeze_pretrained(self):
        # Freeze all parameters except for the adapters
        for name, param in self.named_parameters():
            if 'ln' not in name and'adapter' not in name and 'cross_attn' not in name:
                param.requires_grad = False

    def forward(self, text_input: Tensor, image_features: Tensor):
        text_input = torch.narrow(text_input, 1, 0, min(text_input.size(1), self.block_size))
        pos = torch.arange(text_input.size(1), dtype=torch.long, device=text_input.device).unsqueeze(0)
        x = self.transformer.wte(text_input) + self.transformer.wpe(pos)
        # Flatten image features patches
        transformed_image_features = image_features.unsqueeze(1).expand(-1, x.size(1), -1)
        for block in self.transformer.h:
            x = block(x, transformed_image_features)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

class Adapter(nn.Module):
    def __init__(self, cfg):
        super(Adapter, self).__init__()
        self.down = nn.Linear(cfg.n_embd, 256)
        self.up = nn.Linear(256 , cfg.n_embd)
        
    def forward(self, x):
        x_down = self.down(x)
        x_gelu = F.gelu(x_down)
        x_up = self.up(x_gelu)
        return x + x_up

class PrefixTuning(nn.Module):
    def __init__(self, cfg, prefix_length):
        super().__init__()
        self.prefix_length = prefix_length
        self.prefixes = nn.Parameter(torch.randn(cfg.n_layer, 2, cfg.n_head, self.prefix_length, cfg.n_embd // cfg.n_head))

    def forward(self, layer_id, batch_size):
        return self.prefixes[layer_id].expand(-1, -1, -1, batch_size, -1).permute(3, 1, 2, 0, 4)
    

#####################    
class Decoder_Prefix(nn.Module):
    def __init__(self, cfg, prefix_length):
        super().__init__()
        self.prefix_tuning = PrefixTuning(cfg, prefix_length)
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            # h =  nn.Sequential(
            #     *[Block(cfg) for _ in range(cfg.n_layer - 6)], 
            #     *[Block(cfg, add_adapter=True) for _ in range(6)]  
            # ),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # load checkpoint if available
        if self.cfg.checkpoint is not None:
            # print('Loading checkpoint', self.cfg.checkpoint)
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)
    
    def freeze_pretrained(self):
        # Freeze all parameters except for the adapters
        for name, param in self.named_parameters():
            if 'ln' not in name and'adapter' not in name and 'cross_attn' not in name:
                param.requires_grad = False

    def forward(self, text_input: Tensor, image_features: Tensor):
        text_input = torch.narrow(text_input, 1, 0, min(text_input.size(1), self.block_size))
        pos = torch.arange(text_input.size(1), dtype=torch.long, device=text_input.device).unsqueeze(0)
        x = self.transformer.wte(text_input) + self.transformer.wpe(pos)
        # Flatten image features patches
        transformed_image_features = image_features.unsqueeze(1).expand(-1, x.size(1), -1)
        for i, block in enumerate(self.transformer.h):
            # Get the prefix for the current layer
            prefix = self.prefix_tuning(i, text_input.size(0))
            
            # Prepend the prefix to the input sequence
            extended_input = torch.cat([prefix, x], dim=2)

            x = block(x, transformed_image_features)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits