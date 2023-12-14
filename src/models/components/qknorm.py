import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
import torch.nn.functional as F

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

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.project = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.attn_weights = IdentityLayer()

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x).chunk(3, dim=-1)
        qkv = [part.reshape(b, n, h, -1).transpose(1, 2) for part in qkv]
        
        q, k, v = qkv

        # Normalize q and k using p=2 norm
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)

        # Use the PyTorch 2.0 function for scaled dot-product attention
        attn = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p)
        attn = self.attn_weights(attn)
        
        # Transpose and reshape the attention output to combine the heads
        out = attn.transpose(1, 2).reshape(b, n, -1)
        
        return self.project(out)



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
        x = self.net(x)
        return 0.5 * x + 0.5 * x.mean(dim=1, keepdim=True)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size=224, patch_size=16, num_classes=10, dim=768, depth=12, heads=8, mlp_dim=3072, channels=3, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim, channels)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
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
