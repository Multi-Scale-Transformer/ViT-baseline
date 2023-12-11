import sys
root = '/root/workspace/lightning-hydra-template'
sys.path.append(root)


import cv2

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

def visualize_attention_weights(extractor, img_path, save_path='attention_map.png', alpha=0.8, contrast_factor=0.5):
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Open the image file
    img_ori = Image.open(img_path).convert("RGB")
    img = test_transforms(img_ori)
    img = img.unsqueeze(0)
    
    # Assume 'device' is defined somewhere
    # e.g., device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = img.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        attention_maps = extractor(img)
    
    attn_weights = attention_maps['attn_weights']
    num_layers = len(attn_weights)
    
    # New subplot layout with labels in the first column
    fig, axes = plt.subplots(nrows=3, ncols=num_layers+1, figsize=((num_layers+1) * 3, 9), dpi=100)
    
    # Set labels for the rows
    row_labels = ['Attention Map', 'cls_token', 'Ori Image']
    
    for col, feature in enumerate(attn_weights, start=2):  # Start from the third column
        attn_matrix = feature['output'].squeeze(0)
        attn_matrix = attn_matrix / attn_matrix.sum(dim=-1, keepdim=True)
        attn_matrix = attn_matrix.to(dtype=torch.float32).cpu().numpy()
        
        # Normalize cls_attn to enhance contrast
        cls_attn = attn_matrix[0, 1:].reshape(14, 14)
        min_val = cls_attn.min()
        max_val = cls_attn.max()
        cls_attn = (cls_attn - min_val) / (max_val - min_val)  # Normalize to [0, 1]
        cls_attn = contrast_factor + (1 - contrast_factor) * cls_attn  # Scale to [contrast_factor, 1]
        cls_attn_upsampled = np.kron(cls_attn, np.ones((16, 16)))
        
        # Plot the original attention map
        ax = axes[0, col - 1]
        im = ax.imshow(attn_matrix, cmap='coolwarm', aspect='equal')
        ax.axis('off')
        
        # Overlay the upsampled cls_token attention map on the original image
        ax = axes[1, col - 1]
        img_with_heatmap = np.array(img_ori)
        img_with_heatmap = cv2.resize(img_with_heatmap, (224, 224))
        heatmap = np.uint8(255 * cls_attn_upsampled)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img_with_heatmap = cv2.addWeighted(img_with_heatmap, alpha, heatmap, 1 - alpha, 0)
        ax.imshow(img_with_heatmap)
        ax.axis('off')
        
        # Add a cropped original image
        ax = axes[2, col - 1]
        cropped_img = cv2.resize(np.array(img_ori), (224, 224))  # Replace this with actual cropping logic if needed
        ax.imshow(cropped_img)
        ax.axis('off')
    
    # Place the labels in the first column of each row
    for i, label in enumerate(row_labels):
        axes[i, 0].axis('off')  # Turn off axis
        axes[i, 0].text(0.5, 0.5, label, va='center', ha='center', fontsize=16, transform=axes[i, 0].transAxes)
    
    # Adjust the figure layout
    fig.subplots_adjust(right=0.85, hspace=0.2, wspace=0.2)
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_ticks_position('left')
    
    # Save and close the figure
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
    # intermediate_features = extractor(img)

    visualize_attention_weights(extractor, img_path = "/root/SharedData/datasets/imagenette2/val/n01440764/ILSVRC2012_val_00009111.JPEG",
                                )

