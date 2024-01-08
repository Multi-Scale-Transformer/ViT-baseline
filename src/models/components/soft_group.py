import torch
import math
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

def softmax_with_temperature(logits, temperature=1.0):
    """
    Apply softmax with a temperature coefficient.

    Parameters:
    - logits: torch.Tensor, unnormalized log probabilities (often the output of a neural network layer)
    - temperature: float, temperature coefficient, controls the smoothness of the output distribution

    Returns:
    - torch.Tensor, the probabilities after applying softmax with temperature
    """
    # 确保温度大于0，避免除以0或负数
    assert temperature > 0, "Temperature must be positive"

    # 除以温度参数
    scaled_logits = logits / temperature

    # 应用softmax
    probs = F.softmax(scaled_logits, dim=-1)

    return probs


def keep_top_values(input_tensor, ratio=0.2):
    b, n, _ = input_tensor.shape
    k = max(int(n * ratio), 1)

    top_values, top_indices = torch.topk(input_tensor, k, dim=-1)

    mask = torch.zeros_like(input_tensor).scatter_(-1, top_indices, 1).bool()
    ones_tensor = torch.ones_like(input_tensor)
    output_tensor = torch.where(mask, ones_tensor, torch.zeros_like(input_tensor))

    return output_tensor


class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x  # 直接返回输入数据，不做任何改变

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.proj(x)  # (B, C, H, W) -> (B, D, H/P, W/P)
        x = x.flatten(2)  # (B, D, H/P, W/P) -> (B, D, N)
        x = x.transpose(1, 2)  # (B, D, N) -> (B, N, D)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (1, 1, D) -> (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1, D) + (B, N, D) -> (B, N+1, D)
        x += self.pos_embedding
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout, gp_num):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.gp_num = gp_num
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.project = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        # Initialize group parameter (gp)
        self.gp = nn.Linear(dim, gp_num, bias=False)
        # init.kaiming_uniform_(self.gp, a=math.sqrt(5))  # Xavier uniform initialization
        self.gelu = nn.GELU()
        self.alpha = nn.Parameter(torch.randn((1, self.num_heads,1,1)))

    def forward(self, x):
        b, n, c, h = *x.shape, self.num_heads
        qkv = self.qkv(x).chunk(3, dim=-1)
        qkv = [part.reshape(b, n, h, -1).transpose(1, 2) for part in qkv]
        
        q, k, v = qkv
        attn_weights = torch.matmul(q, k.transpose(-1,-2)) * self.scale
        gp = self.gp.weight
        gp = gp.unsqueeze(0).view(self.num_heads, self.gp_num, self.head_dim)

        group_weight = torch.einsum('bhnd,hmd->bhnm', v, gp)
        group_weight = self.gelu((group_weight))
        group_weight = F.softmax(group_weight, dim=-1)
        group_weight = torch.matmul(group_weight, group_weight.transpose(-2,-1))
        
        attn_weights = attn_weights * group_weight
        attn_weights = F.softmax(attn_weights, dim=-1)
        alpha = torch.sigmoid(self.alpha)
        
        attn_weights = (1 - alpha)*attn_weights + alpha*group_weight
        attn_weights = attn_weights / (attn_weights.sum(dim=2, keepdim=True) + 1e-8)
        attn_weights = self.dropout(attn_weights)        
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(b, n, -1)

        
        return self.project(out)



import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftGroupAttention(nn.Module):
    def __init__(self, dim, dropout=0.0, gp_num=49, t=1.0, keep_ratio=1.0):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.keep_ratio = keep_ratio
        self.qkvg = nn.Linear(dim, dim * 4)
        self.project = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gp = nn.Linear(dim, gp_num, bias=False)
        self.t = t

    def forward(self, x):
        b, n, _ = x.shape
        qkvg = self.qkvg(x).chunk(4, dim=-1)

        q, k, v, g = [part.reshape(b, n, -1) for part in qkvg]

        # Original self-attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Group attention weights
        group_weight = self.gp(g)

        group_weight = softmax_with_temperature(group_weight, temperature=self.t)
        group_weight = torch.matmul(group_weight, group_weight.transpose(-2, -1))

        group_weight = keep_top_values(group_weight, self.keep_ratio)


        g_nonzero_ratio = torch.count_nonzero(group_weight) / group_weight.numel()
        attn_weights = attn_scores * group_weight
        a_nonzero_ratio = torch.count_nonzero(attn_weights) / attn_weights.numel()

        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        out = self.project(attn_output)

        return out



class SingleHeadAttention(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5  # Scale factor for the dot products
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.project = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        
        
    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        
        q, k, v = [part.reshape(b, n, -1) for part in qkv]



        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = softmax_with_temperature(attn_scores, 3)


        attn_weights = self.dropout(attn_weights)
        
        # Multiply the attention weights with the values
        attn_output = torch.matmul(attn_weights, v)
        
        # Project the attention output to the original dimension
        out = self.project(attn_output)
        
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout, mlp_dim, gp_num=49, attn_mode='single', t=1.0, keep_ratio=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if attn_mode == 'multi':
            self.attn = MultiHeadAttention(dim, num_heads, dropout, gp_num=gp_num)
        elif attn_mode == 'single_ori':
            self.attn = SingleHeadAttention(dim=dim, dropout=dropout)
        else:
            self.attn = SoftGroupAttention(dim=dim, dropout=dropout, gp_num=gp_num, t=t, keep_ratio=keep_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_dim, dropout, gp_num=49, attn_mode='multi', t=1.0, keep_ratio=1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            # gp_num = 192 - (182 // 11) * i
            keep_ratio = 1.0 - (0.8 / 11) * i
            self.layers.append(TransformerBlock(dim, num_heads, dropout, mlp_dim=mlp_dim, gp_num=gp_num, attn_mode=attn_mode,t=t, keep_ratio=keep_ratio))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size=224, patch_size=16, num_classes=10, dim=192, depth=12, heads=3, mlp_dim=768, channels=3, dropout=0.0, gp_num=49, attn_mode='single',t=1.0):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim, channels)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, gp_num=gp_num, attn_mode=attn_mode,t=t)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = x[:, 0]  # Take the CLS token
        x = self.to_latent(x)
        return self.mlp_head(x)

if __name__ == '__main__':
    model = ViT()
    img = torch.randn(1, 3, 224, 224)
    preds = model(img)
    print(preds.shape)  # (1, num_classes)
