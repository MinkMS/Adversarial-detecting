import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import os

# ========= SQUEEZING METHODS =========
def reduce_bit_depth(img_tensor, bits=4):
    x = img_tensor.clone()
    levels = 2 ** bits
    x = torch.floor(x * levels) / levels
    return x

def median_filter(img_tensor):
    img = transforms.ToPILImage()(img_tensor.cpu())
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return transforms.ToTensor()(img).to(img_tensor.device)

def rgb_channel_squeeze(img_tensor, bits=4):
    squeezed = []
    for c in img_tensor:
        levels = 2 ** bits
        c = torch.floor(c * levels) / levels
        squeezed.append(c)
    return torch.stack(squeezed).to(img_tensor.device)

# ========= FEATURE EXTRACTION =========
def entropy(p):
    p = torch.clamp(p, 1e-10, 1.0)
    return -(p * torch.log(p)).sum(dim=1)

def extract_features(model, x, squeezers, squeezer_names):
    with torch.no_grad():
        p_orig = torch.softmax(model(x.unsqueeze(0)), dim=1)

    features = {}
    for squeezer, name in zip(squeezers, squeezer_names):
        x_sq = squeezer(x)
        with torch.no_grad():
            p_sq = torch.softmax(model(x_sq.unsqueeze(0)), dim=1)
        conf_diff = (p_orig.max(1).values - p_sq.max(1).values).item()
        kl = F.kl_div(p_sq.log(), p_orig, reduction='batchmean').item()
        changed = int(p_orig.argmax(1) != p_sq.argmax(1))
        ent_diff = (entropy(p_sq) - entropy(p_orig)).item()
        features[f'{name}_conf_drop'] = conf_diff
        features[f'{name}_kl'] = kl
        features[f'{name}_changed'] = changed
        features[f'{name}_entropy'] = ent_diff
    return features

# Script chứa funtion squeeze ảnh theo paper
