import os
import random
import torch
import joblib
import numpy as np
from torchvision import transforms, models
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchattacks import fgsm_attack, pgd_attack, bim_attack
from Squeeze_func_paper import extract_features as extract_paper
from Squeeze_func_me import extract_features as extract_me

# ==== CONST ====
DATA_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\Food101_unused_half\clean"
IMG_SIZE = 512
EPSILON = 4/255
ALPHA = 2/255
STEPS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== TRANSFORMS ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# ==== LOAD MODEL ====
def load_model(name):
    if name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 101)
        model.load_state_dict(torch.load("food101_resnet18.pth", map_location=DEVICE))
    elif name == "efficientnet_b0":
        from timm import create_model
        model = create_model("efficientnet_b0", pretrained=False, num_classes=101)
        model.load_state_dict(torch.load("efficientnet_food101.pth", map_location=DEVICE))
    else:
        raise ValueError("Invalid model name.")
    return model.to(DEVICE).eval()

# ==== SELECT RANDOM IMAGE ====
def pick_random_image():
    classes = os.listdir(DATA_DIR)
    cls = random.choice(classes)
    img_path = random.choice(os.listdir(os.path.join(DATA_DIR, cls)))
    full_path = os.path.join(DATA_DIR, cls, img_path)
    return full_path, cls

# ==== APPLY ATTACK ====
def apply_attack(model, x, y, kind):
    if kind == "fgsm":
        return fgsm_attack(model, x, y, EPSILON)
    elif kind == "pgd":
        return pgd_attack(model, x, y, EPSILON, ALPHA, STEPS)
    elif kind == "bim":
        return bim_attack(model, x, y, EPSILON, ALPHA, STEPS)
    else:
        raise ValueError("Invalid attack type.")

# ==== MAIN ====
def main():
    clf_name = input("Choose classification model [resnet18 / efficientnet_b0]: ").strip()
    attack_type = input("Choose attack method [fgsm / pgd / bim]: ").strip()
    rf_type = input("Choose RF detector model [paper / me]: ").strip()

    print(f"\nLoading classification model: {clf_name}")
    clf_model = load_model(clf_name)

    print(f"Loading RF detector model: {rf_type}")
    rf = joblib.load(f"rf_detector_{rf_type}.pkl")

    path, true_class = pick_random_image()
    print(f"\nImage picked: {os.path.basename(path)} | True class: {true_class}")

    img_raw = read_image(path).float() / 255.
    x_input = transform(to_pil_image(img_raw)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = clf_model(x_input).argmax(1).item()

    print(f"Classification model prediction: {pred} ({clf_name})")

    label_tensor = torch.tensor([pred]).to(DEVICE)
    x_adv = apply_attack(clf_model, x_input, label_tensor, attack_type)

    print(f"Attack applied: {attack_type.upper()}")

    # Extract squeeze features
    img_clean_np = inv_normalize(x_input.squeeze().cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
    img_adv_np = inv_normalize(x_adv.squeeze().detach().cpu()).clamp(0, 1).permute(1, 2, 0).numpy()

    if rf_type == "paper":
        feat_clean = extract_paper(img_clean_np)
        feat_adv = extract_paper(img_adv_np)
    else:
        feat_clean = extract_me(img_clean_np)
        feat_adv = extract_me(img_adv_np)

    feat_diff = np.abs(np.array(feat_clean) - np.array(feat_adv)).reshape(1, -1)

    result = rf.predict(feat_diff)[0]
    label = "DEFECTED" if result == 1 else "CLEAN"

    print(f"\nRandom Forest DETECTOR result: {label}")
    print("Done!")

if __name__ == "__main__":
    main()
