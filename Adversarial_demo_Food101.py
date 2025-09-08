import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from timm import create_model
import torchattacks

# ===== CONFIG =====
IMG_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\Food-101\images"
IMG_SIZE = 512
EPSILON = 10 / 255
ALPHA = 10 / 255
STEPS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== LOAD CLASS LABELS =====
def load_classes():
    return [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]

CLASS_NAMES = load_classes()

# ===== TRANSFORMS =====
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

# ===== LOAD RANDOM IMAGE =====
def load_random_image():
    while True:
        cls = random.choice(CLASS_NAMES)
        cls_path = os.path.join(IMG_DIR, cls)
        img_candidates = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".png"))]
        if img_candidates:
            break

    img_file = random.choice(img_candidates)
    img_path = os.path.join(cls_path, img_file)
    img = read_image(img_path).float() / 255.
    return transform(to_pil_image(img)).unsqueeze(0), cls, img_path

# ===== LOAD MODELS =====
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

# ===== APPLY ATTACK =====
def apply_attack(model, x, y, attack_type):
    if attack_type == "fgsm":
        atk = torchattacks.FGSM(model, eps=EPSILON)
    elif attack_type == "bim":
        atk = torchattacks.BIM(model, eps=EPSILON, alpha=ALPHA, steps=STEPS)
    elif attack_type == "pgd":
        atk = torchattacks.PGD(model, eps=EPSILON, alpha=ALPHA, steps=STEPS)
    else:
        raise ValueError("Invalid attack type")
    return atk(x, y)

# ===== MAIN FUNCTION =====
def main():
    model_name = input("Chọn model (resnet18 / efficientnet_b0): ").strip().lower()
    attack_choice = input("Chọn attack (fgsm / bim / pgd / all): ").strip().lower()

    if model_name not in ["resnet18", "efficientnet_b0"]:
        print("Model không hợp lệ.")
        return
    if attack_choice not in ["fgsm", "bim", "pgd", "all"]:
        print("Loại attack không hợp lệ.")
        return

    attack_list = ["fgsm", "bim", "pgd"] if attack_choice == "all" else [attack_choice]

    x, true_cls, img_path = load_random_image()
    print(f"Selected image: {img_path} | True label: {true_cls}")

    model = load_model(model_name)
    x = x.to(DEVICE)

    with torch.no_grad():
        y = model(x).argmax(1)

    preds = {"clean": CLASS_NAMES[y.item()]}
    images = {"clean": x}

    for attack in attack_list:
        adv = apply_attack(model, x, y, attack)
        with torch.no_grad():
            output = model(adv)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(1)
            confidence = probs[0, pred].item()
        preds[attack] = f"{CLASS_NAMES[pred.item()]} ({confidence:.2%})"
        images[attack] = adv

    # ====== PLOT (clean on top, attacks below) ======
    num_attacks = len(attack_list)
    fig_width = max(4 * num_attacks, 5)
    fig, axes = plt.subplots(2, num_attacks, figsize=(fig_width, 6), gridspec_kw={'height_ratios': [1, 1]})

    # If only one attack, axes may not be 2D, force reshape
    if num_attacks == 1:
        axes = axes.reshape(2, 1)

    # Plot CLEAN on top row, center column
    clean_img = inv_transform(images["clean"].squeeze().detach().cpu()).clamp(0, 1)
    for ax in axes[0]:
        ax.axis("off")
    center_col = num_attacks // 2
    axes[0, center_col].imshow(clean_img.permute(1, 2, 0).numpy())
    axes[0, center_col].set_title(f"CLEAN\n{preds['clean']}", fontsize=10)

    # Plot attacks on bottom row
    for i, attack in enumerate(attack_list):
        img = inv_transform(images[attack].squeeze().detach().cpu()).clamp(0, 1)
        axes[1, i].imshow(img.permute(1, 2, 0).numpy())
        axes[1, i].set_title(f"{attack.upper()}\n{preds[attack]}", fontsize=10)
        axes[1, i].axis('off')

    plt.suptitle(f"Model: {model_name} | True: {true_cls}", fontsize=14)
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("attack_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()

# ===== RUN =====
if __name__ == "__main__":
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {DEVICE}")
    main()

# Script thực hiện FGSM, BIM và PGD trên ảnh random từ Food101, hiển thị hình ảnh gốc và các hình ảnh bị tấn công, cùng với các dự đoán