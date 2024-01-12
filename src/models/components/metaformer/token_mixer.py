import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SoftgroupAttention(nn.Module):
    """
    Modified Softgroup Attention incorporating elements from MultiHeadAttention in Code B.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        gp_num = 20
        # Group parameter and additional functions
        self.gp_num = gp_num
        self.gp = nn.Linear(dim, gp_num, bias=False)
        self.t = nn.Parameter(torch.ones(1))
        self.softplus = nn.Softplus()

    def forward(self, x):
        B, H, W, C = x.shape

        N = H * W
        qkv = self.qkv(x).chunk(3, dim=-1)
        qkv = [part.reshape(B, N, self.num_heads, -1).transpose(1, 2) for part in qkv]

        q, k, v = qkv
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        gp = self.gp.weight
        gp = gp.unsqueeze(0).view(self.num_heads, self.gp_num, self.head_dim)

        group_weight = torch.einsum('bhnd,hmd->bhnm', q, gp)
        t = self.softplus(self.t)
        group_weight = softmax_with_temperature(group_weight, temperature=t)
        group_weight = torch.matmul(group_weight, group_weight.transpose(-2, -1))

        attn_weights = attn_weights * group_weight
        attn_weights = attn_weights / (attn_weights.sum(dim=2, keepdim=True) + 1e-8)
        attn_weights = self.attn_drop(attn_weights)

        x = torch.matmul(attn_weights, v)
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class HardgroupAttention(nn.Module):
    """
    Modified Softgroup Attention incorporating elements from MultiHeadAttention in Code B.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        gp_num = 20
        # Group parameter and additional functions
        self.gp_num = gp_num
        self.gp = nn.Linear(dim, gp_num, bias=False)


    def forward(self, x):
        B, H, W, C = x.shape

        N = H * W
        qkv = self.qkv(x).chunk(3, dim=-1)
        qkv = [part.reshape(B, N, self.num_heads, -1).transpose(1, 2) for part in qkv]

        q, k, v = qkv
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        gp = self.gp.weight
        gp = gp.unsqueeze(0).view(self.num_heads, self.gp_num, self.head_dim)

        group_weight = torch.einsum('bhnd,hmd->bhnm', q, gp)
        _, idx = torch.topk(group_weight, k=1, dim=-1)
        group_weight = torch.zeros_like(group_weight)
        group_weight.scatter_(dim=-1, index=idx, value=1)
        group_weight = torch.matmul(group_weight, group_weight.transpose(-2, -1))
        # 16 32 60 49
        # keep_num = min(C//4, N)
        

        attn_weights = attn_weights * group_weight
        attn_weights = attn_weights / (attn_weights.sum(dim=2, keepdim=True) + 1e-8)
        attn_weights = self.attn_drop(attn_weights)

        x = torch.matmul(attn_weights, v)
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == '__main__':
    model = SoftgroupAttention(dim=64)

    img = torch.randn(1, 56, 56, 64)
    preds = model(img)
    print(preds.shape)