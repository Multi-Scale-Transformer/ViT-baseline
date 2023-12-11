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

def visualize_attention_weights(extractor, img_path, save_path='attention_map.png'):
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to common size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
    ])
    
    img = Image.open(img_path)
    # input = Image.new('RGB', (448, 448), (255, 255, 255))
    img = test_transforms(img)
    img = img.unsqueeze(0)  # 添加batch维度
    
    img = img.to(device=device, dtype=torch.bfloat16)
    # 将图像输入模型以收集注意力权重
    with torch.no_grad():
        attention_maps = extractor(image)
    
    # 获取attn_weights部分
    attn_weights = attention_maps['attn_weights']
    num_layers = len(attn_weights)
    
    # 创建一个大图，以便存储所有的热力图
    fig, axes = plt.subplots(nrows=1, ncols=num_layers, figsize=(num_layers * 3, 3), dpi=100)
    
    # 遍历每层的注意力权重
    for col, feature in enumerate(attn_weights):
        # 归一化注意力权重
        attn_matrix = feature['output'].squeeze(0)  # 假设 batch size 是 1
        attn_matrix = attn_matrix / attn_matrix.sum(dim=-1, keepdim=True)
        attn_matrix = attn_matrix.to(dtype=torch.float32).cpu().numpy()

        # 绘制热力图
        ax = axes[col]
        im = ax.imshow(attn_matrix, cmap='coolwarm', aspect='auto')
        
        # 关闭坐标轴提升清晰度
        ax.axis('off')
        
        # 在子图上标注层数
        ax.set_title(f'Layer {col+1}', fontsize=14, color='red')
    
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
        self.attn_weights_features = []

        # Register a hook for each layer of interest
        for name, layer in self.model.named_modules():
            if 'attn_weights' in name:
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
        self.attn_weights_features.clear()

        _ = self.model(x)  # Perform the model's forward pass

        # Return a dictionary with separate feature lists
        return {
            'attn_weights': self.attn_weights_features
        }

if __name__ == '__main__':
    device = torch.device('cuda:7')
    model = NetteLitModule.load_from_checkpoint('/root/workspace/lightning-hydra-template/logs/train/multiruns/2023-12-10_16-32-08/15/checkpoints/epoch_044.ckpt', map_location=device)
    model = model.net
    model.eval()
    extractor = IntermediateFeaturesExtractor(model)
    


    # Forward pass through the extractor, which will collect the intermediate features
    intermediate_features = extractor(img)

    visualize_attention_weights(extractor, img_path = "/root/SharedData/datasets/imagenette2/val/n01440764/ILSVRC2012_val_00009111.JPEG",
                                image=img)

