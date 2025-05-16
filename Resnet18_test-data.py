import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torchattacks
from PIL import Image
from tqdm import tqdm

# ========== CONFIG ==========
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 128
OUTPUT_DIR = 'Adversarial-Example'
EPS = 8 / 255
ALPHA = 2 / 255
PGD_STEPS = 10
NUM_SAMPLES = 2000  # <= điều chỉnh số ảnh test để sinh, ví dụ 2000 thôi cho demo nhanh

# ========== LOAD MODEL ==========
model = resnet18(num_classes=10)
model.load_state_dict(torch.load(r'C:\Users\Mink\OneDrive\Documents\GitHub\Adversarial-detecting\resnet18_128x128.pth', map_location=DEVICE))
model = model.to(DEVICE).eval()

# ========== LOAD CIFAR-10 TEST SET ==========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
class_names = testset.classes

# ========== DEFINE ATTACKS ==========
fgsm = torchattacks.FGSM(model, eps=EPS)
pgd = torchattacks.PGD(model, eps=EPS, alpha=ALPHA, steps=PGD_STEPS)

# ========== CREATE FOLDER STRUCTURE ==========
def make_dirs():
    for split in ['clean', 'defected/fgsm', 'defected/pgd']:
        for cls in class_names:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)
make_dirs()

# ========== PROCESS IMAGES ==========
for idx, (img, label) in tqdm(enumerate(testset), total=NUM_SAMPLES, desc="Generating structured dataset"):
    if idx >= NUM_SAMPLES:
        break

    class_name = class_names[label]
    img_tensor = img.unsqueeze(0).to(DEVICE)  # [1,3,H,W]
    dummy_label = torch.tensor([label]).to(DEVICE)

    # Convert to PIL
    img_pil = transforms.ToPILImage()(img)
    img_pil.save(f"{OUTPUT_DIR}/clean/{class_name}/img_{idx:05}.png", optimize=True)

    # FGSM
    adv_fgsm = fgsm(img_tensor, dummy_label)
    adv_fgsm_pil = transforms.ToPILImage()(adv_fgsm.squeeze().cpu().clamp(0,1))
    adv_fgsm_pil.save(f"{OUTPUT_DIR}/defected/fgsm/{class_name}/img_{idx:05}.png", optimize=True)

    # PGD
    adv_pgd = pgd(img_tensor, dummy_label)
    adv_pgd_pil = transforms.ToPILImage()(adv_pgd.squeeze().cpu().clamp(0,1))
    adv_pgd_pil.save(f"{OUTPUT_DIR}/defected/pgd/{class_name}/img_{idx:05}.png", optimize=True)
