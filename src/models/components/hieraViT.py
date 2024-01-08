import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W) -> (B, D, H/P, W/P)
        x = x.flatten(2)  # (B, D, H/P, W/P) -> (B, D, N)
        x = x.transpose(1, 2)  # (B, D, N) -> (B, N, D)
        x += self.pos_embedding
        return x

class Downsample(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.reduction = nn.Conv2d(dim, dim_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).view(B, D, H, W)
        x = self.reduction(x)
        x = x.flatten(2).transpose(1, 2)
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

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x).chunk(3, dim=-1)
        qkv = [part.reshape(b, n, h, -1).transpose(1, 2) for part in qkv]
        
        q, k, v = qkv


        # Use the PyTorch 2.0 function for scaled dot-product attention
        attn = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p)
        
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
        return self.net(x)

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

class TransformerStage(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class HierarchicalViT(nn.Module):
    def __init__(self, *, image_size=224, num_classes=10, channels=3, dropout=0.1):
        super().__init__()
        stages = [2, 2, 6, 2]
        dims = [40, 80, 160, 320]
        patch_sizes = [4, 2, 2, 2]
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, (dim, stage, patch_size) in enumerate(zip(dims, stages, patch_sizes)):
            if i == 0:
                self.stages.append(PatchEmbedding(image_size, patch_size, dim, channels))
            else:
                self.stages.append(TransformerStage(dim, stage, 8, dim * 4, dropout))
                self.downsamples.append(Downsample(dims[i-1], dim))
            image_size //= patch_size

        self.norm = nn.LayerNorm(dims[-1])
        self.mlp_head = nn.Sequential(
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        x = self.norm(x.mean(dim=1))  # Global average pooling
        return self.mlp_head(x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierarchicalViT().to(device)
    model.eval()
    inputs = torch.randn(1, 3, 224, 224).to(device)
    outputs = model(inputs)
    print(outputs)
