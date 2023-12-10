import sys
root = '/root/workspace/lightning-hydra-template'
sys.path.append(root)



from src.models.components.soft_group import ViT
from src.models.nette_module import NetteLitModule
import torch
from torchinfo import summary
from torch import nn
from collections.abc import Iterable  # for Python 3.3 and above
from typing import Callable, Dict
from torchvision import transforms
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F

# 添加字体路径
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=12)  # 可以指定字体大小

def visualize_attention_weights(extractor, image, save_path='attention_map.png'):
    # 将图像输入模型以收集注意力权重
    with torch.no_grad():
        attention_maps = extractor(image)
    
    # 创建一个大图，以便存储所有的热力图
    num_layers = len(next(iter(attention_maps.values())))
    num_heads = attention_maps[next(iter(attention_maps))][0]['output'].size(0)
    fig, axes = plt.subplots(nrows=3, ncols=num_layers, figsize=(num_layers * 3, 3 * 3), dpi=100)
    
    # 定义每行代表的矩阵类型
    row_labels = ['attn_scores', 'group_weight', 'attn_weights']
    
    # 遍历每种权重和每层的权重
    for row, (key, feature_list) in enumerate(attention_maps.items()):
        # 在最左侧添加矩阵类型的注释，使用指定的字体
        axes[row, 0].text(-0.5, 0.5, row_labels[row], va='center', ha='right',
                           fontsize=12, fontproperties=font_prop, transform=axes[row, 0].transAxes)
        
        for col, feature in enumerate(feature_list):
            # 归一化注意力权重
            attn_matrix = feature['output'].squeeze(0)  # 假设 batch size 是 1
            attn_matrix = attn_matrix / attn_matrix.sum(dim=-1, keepdim=True)
            attn_matrix = attn_matrix.to(dtype=torch.float32).cpu().numpy()
            
            # 绘制热力图
            ax = axes[row, col]
            im = ax.imshow(attn_matrix, cmap='coolwarm', aspect='auto')
            
            # 仅在顶部的子图上标注层数，使用指定的字体
            if row == 0:
                ax.set_title(f'Layer {col+1}', fontsize=14, fontproperties=font_prop, color='red')
            
            # 关闭坐标轴提升清晰度
            ax.axis('off')
    
    # 调整布局和添加颜色条
    fig.subplots_adjust(right=0.85, hspace=0.2, wspace=0.2)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)  # 设置颜色条刻度字体大小
    cbar.ax.yaxis.set_ticks_position('left')  # 将颜色条刻度放置到左侧
    
    # 保存可视化结果
    plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.close(fig)

def print_module_attr_names(module, prefix=''):
    for name in module.__dict__:
        if not name.startswith('_'):
            print(f"{prefix}{name}")
    for name, child in module.named_children():
        print(f"{prefix}{name}:")
        print_module_attr_names(child, prefix=prefix + '  ')

class IntermediateFeaturesExtractor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.attn_scores_features = []
        self.group_weight_features = []
        self.attn_weights_features = []

        # Register a hook for each layer of interest
        for name, layer in self.model.named_modules():
            if 'attn_scores' in name:
                layer.register_forward_hook(self.save_output_hook('attn_scores', self.attn_scores_features))
            elif 'group_weight' in name:
                layer.register_forward_hook(self.save_output_hook('group_weight', self.group_weight_features))
            elif 'attn_weights' in name:
                layer.register_forward_hook(self.save_output_hook('attn_weights', self.attn_weights_features))

    def save_output_hook(self, feature_type, feature_list):
        def hook(module, input, output):
            # Save the output of the layer to the appropriate list
            feature_list.append({
                "layer": feature_type,
                "output": output.detach()  # Use detach() if you do not plan to backpropagate through here
            })
        return hook

    def forward(self, x: torch.Tensor):
        # Clear the feature lists
        self.attn_scores_features.clear()
        self.group_weight_features.clear()
        self.attn_weights_features.clear()

        _ = self.model(x)  # Perform the model's forward pass

        # Return a dictionary with separate feature lists
        return {
            'attn_scores': self.attn_scores_features,
            'group_weight': self.group_weight_features,
            'attn_weights': self.attn_weights_features
        }

if __name__ == '__main__':
    device = torch.device('cuda:7')
    model = NetteLitModule.load_from_checkpoint('/root/workspace/lightning-hydra-template/logs/train/runs/2023-12-09_16-58-16/checkpoints/epoch_000.ckpt', map_location=device)
    model = model.net
    model.eval()
    
    
    extractor = IntermediateFeaturesExtractor(model)
    
    img = torch.randn(1, 3, 224, 224)
    img = img.to(device=device, dtype=torch.bfloat16)

    # Forward pass through the extractor, which will collect the intermediate features
    intermediate_features = extractor(img)

    visualize_attention_weights(extractor, image=img)

