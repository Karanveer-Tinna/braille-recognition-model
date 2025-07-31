import os
import cv2 as cv
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

# ----------- Preprocessing Function -----------
def preprocess_image(img: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    denoised = cv.bilateralFilter(equalized, d=9, sigmaColor=100, sigmaSpace=100)
    thresh = cv.adaptiveThreshold(
        denoised, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
        blockSize=5, C=2
    )
    return thresh

# ----------- CNN Model -----------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=26):  # Assume A–Z Braille classification
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ----------- Load YOLO and Prepare Dataset -----------
def load_yolo_dataset(image_dir, label_dir, transform):
    data = []
    labels = []

    label_map = {chr(i + 65): i for i in range(26)}  # A-Z mapping

    for label_file in tqdm(os.listdir(label_dir)):
        image_file = os.path.splitext(label_file)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        image = cv.imread(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w, h = map(float, parts)
                x1 = int((x_center - w / 2) * width)
                y1 = int((y_center - h / 2) * height)
                x2 = int((x_center + w / 2) * width)
                y2 = int((y_center + h / 2) * height)

                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, width), min(y2, height)

                cropped = image[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                processed = preprocess_image(cropped)
                processed = cv.resize(processed, (28, 28))
                processed = transform(processed).squeeze(0)

                data.append(processed)
                labels.append(int(class_id))

    data_tensor = torch.stack(data)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(data_tensor, label_tensor)

# ----------- Main Training/Eval Script -----------
def main():
    train_img_dir = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\images\train"
    train_lbl_dir = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\labels\train"

    test_img_dir = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\images\test"
    test_lbl_dir = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\labels\test"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("[INFO] Loading training data...")
    train_dataset = load_yolo_dataset(train_img_dir, train_lbl_dir, transform)
    print("[INFO] Loading testing data...")
    test_dataset = load_yolo_dataset(test_img_dir, test_lbl_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=26).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    print("[INFO] Training...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X.unsqueeze(1))  # (B, 1, 28, 28)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Evaluation
    print("[INFO] Evaluating...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X.unsqueeze(1))
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"[RESULT] Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
