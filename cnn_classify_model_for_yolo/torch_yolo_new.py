import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from pathlib import Path
import sys

# --- CNN Model ---
class BrailleCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(BrailleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 28 → 14
        x = self.pool(torch.relu(self.conv2(x)))  # 14 → 7
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# --- Preprocessing ---
def preprocess_image(img: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blurred = cv.GaussianBlur(equalized, (5,5), 0)
    denoised = cv.bilateralFilter(blurred, 9, 100, 100)
    thresh = cv.adaptiveThreshold(denoised, 255,
                                   cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY_INV,
                                   blockSize=5, C=2)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    return opened

# --- Custom Dataset ---
class YOLOBrailleDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.count = 0
        self.image_paths = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        h, w = image.shape[:2]

        label_path = self.label_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            return None

        Xs, Ys = [], []
        with open(label_path) as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 5:
                    continue
                class_id, x, y, bw, bh = map(float, values)
                xmin = int((x - bw / 2) * w)
                ymin = int((y - bh / 2) * h)
                xmax = int((x + bw / 2) * w)
                ymax = int((y + bh / 2) * h)

                if xmin < 0 or ymin < 0 or xmax > w or ymax > h or xmin >= xmax or ymin >= ymax:
                    continue

                cropped = image[ymin:ymax, xmin:xmax]
                if cropped.size == 0:
                    continue

                processed = preprocess_image(cropped)
                if self.transform:
                    try:
                        processed = self.transform(processed)
                    except Exception as e:
                        continue

                if processed.shape != (1, 28, 28):
                    continue

                Xs.append(processed)
                Ys.append(int(class_id))

        if len(Xs) == 0 or len(Ys) == 0:
            self.count += 1
            return None

        return torch.stack(Xs), torch.tensor(Ys)


# --- Collate function ---
def collate_fn(batch):
    X, Y = [], []
    for b in batch:
        if b is None or len(b) != 2:
            continue
        xb, yb = b
        if xb is None or yb is None or len(xb) == 0:
            continue
        X.extend(xb)
        Y.extend(yb)

    if len(X) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.stack(X), torch.tensor(Y)


# --- Directories ---
# !!! IMPORTANT: DOUBLE-CHECK THAT THESE PATHS ARE CORRECT !!!
train_img_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images"
train_lbl_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\labels"

valid_img_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\valid\images"
valid_lbl_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\valid\labels"

test_img_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\test\images"
test_lbl_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\test\labels"

# --- Transform ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# --- Load Datasets ---
train_dataset = YOLOBrailleDataset(train_img_dir, train_lbl_dir, transform)
valid_dataset = YOLOBrailleDataset(valid_img_dir, valid_lbl_dir, transform)
test_dataset  = YOLOBrailleDataset(test_img_dir, test_lbl_dir, transform)

# --- FIX: Add validation check for dataset lengths ---
if len(train_dataset) == 0:
    print(f"Error: No images (.jpg or .png) found in the training directory: '{train_img_dir}'")
    print("Please ensure the path is correct and the directory contains image files.")
    sys.exit(1) # Exit the script

if len(valid_dataset) == 0:
    print(f"Warning: No images found in the validation directory: '{valid_img_dir}'")

if len(test_dataset) == 0:
    print(f"Warning: No images found in the test directory: '{test_img_dir}'")


# --- Create DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, collate_fn=collate_fn)


# --- Model, Loss, and Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrailleCNN(num_classes=26).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training and Evaluation Functions ---
def train(model, loader, optimizer, criterion):
    model.train()
    total, correct, loss_val = 0, 0, 0
    for X, Y in loader:
        if X.shape[0] == 0: continue
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        loss_val += loss.item()
        _, predicted = outputs.max(1)
        total += Y.size(0)
        correct += predicted.eq(Y).sum().item()

    if total == 0: return 0, 0
    return loss_val / len(loader), 100 * correct / total


def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for X, Y in loader:
            if X.shape[0] == 0: continue
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            _, predicted = outputs.max(1)
            total += Y.size(0)
            correct += predicted.eq(Y).sum().item()
    if total == 0: return 0
    return 100 * correct / total


# --- Run Training ---
epochs = 100
for epoch in range(epochs):
    loss, acc = train(model, train_loader, optimizer, criterion)
    val_acc = evaluate(model, valid_loader)
    print(f"Epoch {epoch+1:02d}: Train Loss={loss:.4f}, Train Acc={acc:.2f}%, Valid Acc={val_acc:.2f}%")

# --- Final Test Accuracy ---
test_acc = evaluate(model, test_loader)
print(f"\nTest Accuracy: {test_acc:.2f}%")

torch.save(model.state_dict(), "yolo_classify_model_new_preprocess.pt")
print("Model saved successfully!")