import os
import random
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from timm import create_model
import torchattacks
from concurrent.futures import ThreadPoolExecutor

# ========== FLAGS ==========
SKIP_CLEAN = True
SKIP_RESNET18_TRAIN = True
SKIP_RESNET18_VAL = False
SKIP_EFFICIENTNET_TRAIN = False
SKIP_EFFICIENTNET_VAL = False

# ========== CONFIG ========== #
SRC_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\Food-101\images"
DEST_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\Food-101-rf"
IMG_SIZE = 512
SPLIT_RATIO = 0.5
ATTACKS = ['fgsm', 'bim', 'pgd']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPSILON = 8 / 255
ALPHA = 8 / 255
STEPS = 10
BATCH_SIZE = 16
NUM_WORKERS = 4

# ========== TRANSFORMS ========== #
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

inv_transform = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

def save_tensor_image(tensor, save_path):
    img = inv_transform(tensor.squeeze().cpu()).clamp(0, 1)
    img_pil = transforms.ToPILImage()(img)
    img_pil.save(save_path)

# ========== MODEL LOADER ========== #
def load_model(name):
    if name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 101)
        model.load_state_dict(torch.load("food101_resnet18.pth", map_location=DEVICE))
    elif name == "efficientnet_b0":
        model = create_model("efficientnet_b0", pretrained=False, num_classes=101)
        model.load_state_dict(torch.load("efficientnet_food101_best.pth", map_location=DEVICE))
    else:
        raise ValueError("Unsupported model")
    return model.to(DEVICE).eval()

# ========== ATTACK ========== #
def run_attack(model, x, y, attack_type):
    if attack_type == "fgsm":
        atk = torchattacks.FGSM(model, eps=EPSILON)
    elif attack_type == "bim":
        atk = torchattacks.BIM(model, eps=EPSILON, alpha=ALPHA, steps=STEPS)
    elif attack_type == "pgd":
        atk = torchattacks.PGD(model, eps=EPSILON, alpha=ALPHA, steps=STEPS)
    else:
        raise ValueError("Unknown attack type")
    return atk(x, y)

# ========== IMAGE COPY ==========
def copy_clean_image(args):
    img_src, out_name, split = args
    clean_path = os.path.join(DEST_DIR, split, "clean", out_name)
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    if not os.path.exists(clean_path):
        img = Image.open(img_src).convert("RGB")
        img.save(clean_path)

# ========== BATCH ATTACK ==========
def process_batch_defected(model, x_batch, y_batch, out_names, split, model_name):
    for attack in ATTACKS:
        adv_batch = run_attack(model, x_batch, y_batch, attack)
        for i in range(len(out_names)):
            adv_path = os.path.join(
                DEST_DIR, split, "defected", model_name,
                out_names[i].replace(".png", f"-{attack}.png")
            )
            save_tensor_image(adv_batch[i].unsqueeze(0), adv_path)

def process_all_defected(model_name, split, img_paths, out_names, cls_names, index_map):
    model = load_model(model_name)
    for i in tqdm(range(0, len(img_paths), BATCH_SIZE), desc=f"{model_name} - {split}"):
        batch_paths = img_paths[i:i + BATCH_SIZE]
        batch_labels = cls_names[i:i + BATCH_SIZE]
        x_batch, y_batch = [], []
        for img_path, cls in zip(batch_paths, batch_labels):
            img = Image.open(img_path).convert("RGB")
            x = transform(img)
            x_batch.append(x)
            y_batch.append(index_map[cls])
        x_batch = torch.stack(x_batch).to(DEVICE)
        y_batch = torch.tensor(y_batch).to(DEVICE)
        process_batch_defected(model, x_batch, y_batch, out_names[i:i + BATCH_SIZE], split, model_name)

# ========== MAIN ========== #
def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    class_names = sorted(os.listdir(SRC_DIR))
    model_names = ["resnet18", "efficientnet_b0"]

    for split in ["train", "val"]:
        os.makedirs(os.path.join(DEST_DIR, split, "clean"), exist_ok=True)
        for model in model_names:
            os.makedirs(os.path.join(DEST_DIR, split, "defected", model), exist_ok=True)

    index_map = {}
    copy_tasks = []

    for cls in tqdm(class_names, desc="Preparing split tasks"):
        cls_path = os.path.join(SRC_DIR, cls)
        img_files = sorted([f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".png"))])
        random.shuffle(img_files)
        index_map[cls] = len(index_map)
        split_idx = int(len(img_files) * SPLIT_RATIO)
        splits = {
            "train": img_files[:split_idx],
            "val": img_files[split_idx:]
        }

        for split in ["train", "val"]:
            for i, img_file in enumerate(splits[split]):
                img_src = os.path.join(cls_path, img_file)
                out_name = f"{cls}-{i:04d}.png"
                copy_tasks.append((img_src, out_name, split))

    # COPY CLEAN IMAGES
    if not SKIP_CLEAN:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(copy_clean_image, copy_tasks), total=len(copy_tasks), desc="Copying clean images"))

    # RUN ATTACKS
    for model_name in model_names:
        for split in ["train", "val"]:
            should_skip = (
                (model_name == "resnet18" and split == "train" and SKIP_RESNET18_TRAIN) or
                (model_name == "resnet18" and split == "val" and SKIP_RESNET18_VAL) or
                (model_name == "efficientnet_b0" and split == "train" and SKIP_EFFICIENTNET_TRAIN) or
                (model_name == "efficientnet_b0" and split == "val" and SKIP_EFFICIENTNET_VAL)
            )
            if should_skip:
                print(f"[SKIP] {model_name} - {split}")
                continue

            clean_dir = os.path.join(DEST_DIR, split, "clean")
            clean_files = sorted(os.listdir(clean_dir))
            img_paths = [os.path.join(clean_dir, f) for f in clean_files]
            out_names = clean_files
            cls_names = [f.split("-")[0] for f in clean_files]

            process_all_defected(model_name, split, img_paths, out_names, cls_names, index_map)

    print("All done: Clean + Defected dataset generated.")

if __name__ == "__main__":
    if DEVICE == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("Using GPU acceleration")
    else:
        print("Using CPU")
    torch.manual_seed(42)
    random.seed(42)
    main()

# Script chia ảnh thành clean và defected theo từng model