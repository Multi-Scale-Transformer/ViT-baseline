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
        attn_drop=0., proj_drop=0., proj_bias=False, group_mode='single', **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.group_mode = group_mode
        
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

       
        if self.group_mode == 'multi':
            gp = self.gp.weight
            gp = gp.unsqueeze(0).view(self.num_heads, self.gp_num, self.head_dim)
            group_weight = torch.einsum('bhnd,hmd->bhnm', q, gp)
            _, idx = torch.topk(group_weight, k=1, dim=-1)
            group_weight = torch.zeros_like(group_weight)
            group_weight.scatter_(dim=-1, index=idx, value=1)
            group_weight = torch.matmul(group_weight, group_weight.transpose(-2, -1))
        else:
            b, h, n, d = q.shape
            q_all_heads = q.permute(0, 2, 1, 3)
            q_all_heads = q_all_heads.reshape(b, n, h*d)
            group_weight = self.gp(q_all_heads)
            _, idx = torch.topk(group_weight, k=1, dim=-1)
            group_weight = torch.zeros_like(group_weight)
            group_weight.scatter_(dim=-1, index=idx, value=1)
            # group_weight_invert = ~group_weight
            group_weight = torch.matmul(group_weight, group_weight.transpose(-2, -1))
            group_weight = group_weight.unsqueeze(1)
            group_weight = group_weight.expand(b, h//2, n, n)
            invert_group_weight = torch.ones_like(group_weight) - group_weight
            invert_group_weight = invert_group_weight.expand(b, h//2, n, n)
            group_weight = torch.concat((group_weight, invert_group_weight), dim=1)

        attn_weights = attn_weights * group_weight
        attn_weights = attn_weights / (attn_weights.sum(dim=2, keepdim=True) + 1e-8)
        attn_weights = self.attn_drop(attn_weights)

        x = torch.matmul(attn_weights, v)
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class HardgroupAttentionV2(nn.Module):
    """
    Modified Softgroup Attention incorporating elements from MultiHeadAttention in Code B.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, group_mode='multi', **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.group_mode = group_mode
        
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        gp_num = 5120 // dim
        # Group parameter and additional functions
        self.gp_num = gp_num
        self.gp_q = nn.Linear(dim, gp_num, bias=False)
        self.gp_k = nn.Linear(dim, gp_num, bias=False)


    def forward(self, x):
        B, H, W, C = x.shape

        N = H * W
        qkv = self.qkv(x).chunk(3, dim=-1)
        qkv = [part.reshape(B, N, self.num_heads, -1).transpose(1, 2) for part in qkv]

        q, k, v = qkv
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

       
        if self.group_mode == 'multi':
            gp_q = self.gp_q.weight
            gp_q = gp_q.unsqueeze(0).view(self.num_heads, self.gp_num, self.head_dim)
            
            gp_k = self.gp_k.weight
            gp_k = gp_k.unsqueeze(0).view(self.num_heads, self.gp_num, self.head_dim)
            
            group_weight_q = torch.einsum('bhnd,hmd->bhnm', q, gp_q)
            _, idx = torch.topk(group_weight_q, k=1, dim=-1)
            group_weight_q = torch.zeros_like(group_weight_q)
            group_weight_q.scatter_(dim=-1, index=idx, value=1)
            
            group_weight_k = torch.einsum('bhnd,hmd->bhnm', k, gp_k)
            _, idx = torch.topk(group_weight_k, k=1, dim=-1)
            group_weight_k = torch.zeros_like(group_weight_k)
            group_weight_k.scatter_(dim=-1, index=idx, value=1)
            
            
            group_weight = torch.matmul(group_weight_q, group_weight_k.transpose(-2, -1))
        else:
            b, h, n, d = q.shape
            q_all_heads = q.permute(0, 2, 1, 3)
            q_all_heads = q_all_heads.reshape(b, n, h*d)
            group_weight = self.gp(q_all_heads)
            _, idx = torch.topk(group_weight, k=1, dim=-1)
            group_weight = torch.zeros_like(group_weight)
            group_weight.scatter_(dim=-1, index=idx, value=1)
            # group_weight_invert = ~group_weight
            group_weight = torch.matmul(group_weight, group_weight.transpose(-2, -1))
            group_weight = group_weight.unsqueeze(1)
            group_weight = group_weight.expand(b, h//2, n, n)
            invert_group_weight = torch.ones_like(group_weight) - group_weight
            invert_group_weight = invert_group_weight.expand(b, h//2, n, n)
            group_weight = torch.concat((group_weight, invert_group_weight), dim=1)

        attn_weights = attn_weights * group_weight
        attn_weights = attn_weights / (attn_weights.sum(dim=2, keepdim=True) + 1e-8)
        attn_weights = self.attn_drop(attn_weights)

        x = torch.matmul(attn_weights, v)
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
    

    


class Attention_qkv(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
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

        self.q = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_v(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
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

        self.v = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W

        # Process only the 'value' part
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Since we are not using query and key, we can directly use the value
        x = v.transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




if __name__ == '__main__':
    model = Attention_v(dim=64)

    img = torch.randn(2, 56, 56, 64)
    preds = model(img)
    print(preds.shape)