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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F


def visualize_attention_maps(attention_scores,j):
    attention_scores = attention_scores.detach().numpy()  # 转换为NumPy数组
    num_heads = attention_scores.shape[1]
    fig, axs = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle('Attention Maps')

    for i in range(num_heads):
        row = i // 3
        col = i % 3
        attn_map = attention_scores[0, i]
        attn_map = torch.from_numpy(attn_map)
        # cos = torch.matmul(attn_map,attn_map.transpose(-1, -2))
        # cos_mean = torch.mean(cos)
        # cos = cos / torch.max(cos)*255
        ce_all = 0
        for m in range(1370):
            for n in range(1370):
                ce = F.cross_entropy(attn_map[m], attn_map[n])
                ce_all = ce_all + ce
        ce_all = ce_all / (1370^2)
        
        # CE = F.cross_entropy(attn_map, attn_map)
        
        
        axs[row, col].imshow(attn_map, cmap='hot', interpolation='nearest')
        
        rank = 1
        # axs[row, col].set_title(f'Head {i+1} (cos: {cos:.2f})')
        axs[row, col].set_title(f'Head {i+1} ce {ce_all:2f}')
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f'/root/workspace/TransFG/attn_maps/{j}.png')  # 保存图像到当前文件夹下的attention_maps.png
    plt.close()  # 关闭绘图窗口

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
    a = 1

    # for name, param in model.named_parameters():
    #     if 'attn_scores' in name:
    #         print(name, param)
    # print_module_attr_names(model.net)


    
    # for i in range(11):
    #     attn_scores_str = [f"transformer.encoder.layer.{i}.attn.softmax"]
    #     attn_scores = FeatureExtractor(model, layers=last_layer)
        
    #     input = Image.open("/root/SharedData/datasets/CUB_200_2011/data_splited/val/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg")
    #     # input = Image.new('RGB', (448, 448), (255, 255, 255))
    #     input = test_transform(input)
    #     input = input.unsqueeze(0)  # 添加batch维度
    #     # input = torch.randn(3,3,448,448)
    #     features = vit_features(input)
    #     # print({name: output.shape for name, output in features.items()})
    #     item = features[last_layer[0]]

    #     visualize_attention_maps(item,j)
