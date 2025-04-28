import random
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import skimage.metrics

# 1. Load 1 ảnh CIFAR-10
transform = transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

img_tensor, label = dataset[random.randint(0 , len(dataset)-1)]  # lấy ảnh random từ số 1 tới 100
img_np = img_tensor.permute(1,2,0).numpy()

# 2. Upscale ảnh từ 32 lên 128 
img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
img_upscaled = img_pil.resize((128, 128), resample=Image.BICUBIC)
img_upscaled_np = np.array(img_upscaled) / 255.0

# 3. Hiển thị 2 ảnh
fig, axs = plt.subplots(1, 2, figsize=(8,4))
axs[0].imshow(img_np)
axs[0].set_title("Original 32x32")
axs[0].axis('off')
axs[1].imshow(img_upscaled_np)
axs[1].set_title("Upscaled 128x128")
axs[1].axis('off')
plt.show()