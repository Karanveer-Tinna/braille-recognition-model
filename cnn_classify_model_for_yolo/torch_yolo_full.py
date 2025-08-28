import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from pathlib import Path

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
    denoised = cv.bilateralFilter(equalized, 9, 100, 100)
    thresh = cv.adaptiveThreshold(denoised, 255,
                                   cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY_INV,
                                   blockSize=5, C=2)
    return thresh

# --- Custom Dataset ---
class YOLOBrailleDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
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
                    # print(f"Skipping malformed line in {label_path}: {line.strip()}")
                    continue
                class_id, x, y, bw, bh = map(float, line.strip().split())
                xmin = int((x - bw / 2) * w)
                ymin = int((y - bh / 2) * h)
                xmax = int((x + bw / 2) * w)
                ymax = int((y + bh / 2) * h)

                cropped = image[ymin:ymax, xmin:xmax]
                if cropped.size == 0:
                    continue
                processed = preprocess_image(cropped)
                if self.transform:
                    processed = self.transform(processed)
                Xs.append(processed)
                Ys.append(int(class_id))
        return torch.stack(Xs), torch.tensor(Ys)

# --- Collate function ---
def collate_fn(batch):
    X, Y = [], []
    for b in batch:
        if b is None:
            continue
        xb, yb = b
        X.extend(xb)
        Y.extend(yb)
    return torch.stack(X), torch.tensor(Y)

# --- Directories ---
train_img_dir = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\images\train"
train_lbl_dir = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\labels\train"

valid_img_dir = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\images\valid"
valid_lbl_dir = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\labels\valid"

test_img_dir = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\images\test"
test_lbl_dir = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\labels\test"

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, collate_fn=collate_fn)

# --- Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrailleCNN(num_classes=26).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
def train(model, loader):
    model.train()
    total, correct, loss_val = 0, 0, 0
    for X, Y in loader:
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
    return loss_val / len(loader), 100 * correct / total

# --- Evaluation ---
def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            _, predicted = outputs.max(1)
            total += Y.size(0)
            correct += predicted.eq(Y).sum().item()
    return 100 * correct / total

# --- Run Training ---
for epoch in range(10):
    loss, acc = train(model, train_loader)
    val_acc = evaluate(model, valid_loader)
    print(f"Epoch {epoch+1}: Train Loss={loss:.4f}, Train Acc={acc:.2f}%, Valid Acc={val_acc:.2f}%")

# --- Final Test Accuracy ---
test_acc = evaluate(model, test_loader)
print(f"\nTest Accuracy: {test_acc:.2f}%")
