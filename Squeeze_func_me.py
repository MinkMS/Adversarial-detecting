import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image, ImageOps
from scipy.stats import entropy
import io

# === Squeeze functions ===
def reduce_bit_depth(x, bits=4):
    x = x.clone()
    levels = 2 ** bits
    return torch.floor(x * levels) / levels

def median_filter(x):
    x = x.unsqueeze(0)
    x_blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x_blur.squeeze(0)

def rgb_channel_squeeze(x):
    avg = x.mean(dim=0, keepdim=True)
    return avg.repeat(3, 1, 1)

def jpeg_compression(x, quality=30):
    to_pil = ToPILImage()
    to_tensor = ToTensor()
    img_pil = to_pil(x.cpu())
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img_jpeg = Image.open(buffer)
    return to_tensor(img_jpeg)

def resize_squeeze(x, size=64):
    to_pil = ToPILImage()
    to_tensor = ToTensor()
    img_pil = to_pil(x.cpu())
    img_small = img_pil.resize((size, size), resample=Image.BILINEAR)
    img_back = img_small.resize((x.shape[2], x.shape[1]), resample=Image.BILINEAR)
    return to_tensor(img_back)

def hist_equalize(x):
    to_pil = ToPILImage()
    to_tensor = ToTensor()
    img_pil = to_pil(x.cpu())
    r, g, b = img_pil.split()
    r = ImageOps.equalize(r)
    g = ImageOps.equalize(g)
    b = ImageOps.equalize(b)
    img_eq = Image.merge("RGB", (r, g, b))
    return to_tensor(img_eq)

# === Feature extractor ===
def extract_features(model, x, squeezers, squeezer_names):
    x = x.unsqueeze(0)
    x = x.to(next(model.parameters()).device)

    with torch.no_grad():
        out_clean = model(x)
        conf_clean = F.softmax(out_clean, dim=1).squeeze()
        pred_label = torch.argmax(conf_clean).item()

    features = {}
    for squeezer, name in zip(squeezers, squeezer_names):
        x_sq = squeezer(x.squeeze(0)).unsqueeze(0).to(x.device)
        with torch.no_grad():
            out_sq = model(x_sq)
            conf_sq = F.softmax(out_sq, dim=1).squeeze()

        conf_drop = conf_clean[pred_label].item() - conf_sq[pred_label].item()
        kl = F.kl_div(conf_sq.log(), conf_clean, reduction='batchmean').item()
        changed = float(torch.argmax(conf_sq) != pred_label)
        entropy_diff = entropy(conf_clean.cpu().numpy()) - entropy(conf_sq.cpu().numpy())

        features[f"{name}_conf_drop"] = conf_drop
        features[f"{name}_kl"] = kl
        features[f"{name}_changed"] = changed
        features[f"{name}_entropy"] = entropy_diff

    return features

# Script chứa các hàm squeeze ảnh theo feature tự chọn