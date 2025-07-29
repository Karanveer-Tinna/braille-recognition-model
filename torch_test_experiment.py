import cv2 as cv
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from itertools import product

DATASET_DIR = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Dataset"

class BrailleDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = (self.X[idx].squeeze() * 255).astype(np.uint8)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(self.X[idx])
        return img, self.y[idx]

def main():
    X, y = [], []
    label_map = {chr(97 + i): i for i in range(26)}

    for img_path in Path(DATASET_DIR).iterdir():
        processed = preprocess_image(cv.imread(str(img_path)))
        X.append(processed)
        y.append(label_map[img_path.name[0]])

    X = np.array(X).astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
    y = np.array(y).astype(np.int64)

    train_loader, val_loader, test_loader = split_into_loaders(X, y)

    best_model, best_device, best_params = hypertune_model(train_loader, val_loader)

    evaluate_model(test_loader, best_model, best_device)

    print(f"Best Hyperparameters: {best_params}")

    image_path = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\images\test\IMG_4479_jpg.rf.74bfb6d141e15e92aa9aa0cb70236cd5.jpg"
    label_path = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\labels\test\IMG_4479_jpg.rf.74bfb6d141e15e92aa9aa0cb70236cd5.txt"

    predicted = predict_braille_text_from_image(image_path, best_model, best_device)
    actual = get_actual_text_from_yolo_labels(label_path)
    compare_predicted_and_actual(predicted, actual)

def preprocess_image(img: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    morph = cv.morphologyEx(adaptive, cv.MORPH_OPEN, kernel, iterations=1)
    return morph

def split_into_loaders(X: np.array, y: np.array):
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])

    test_transform = transforms.ToTensor()

    train_dataset = BrailleDataset(X_train, y_train, transform=train_transform)
    val_dataset = BrailleDataset(X_val, y_val, transform=test_transform)
    test_dataset = BrailleDataset(X_test, y_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

class BrailleCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 26)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

def hypertune_model(train_loader, val_loader):
    param_grid = {
        "lr": [0.001, 0.0005],
        "dropout": [0.3, 0.5],
        "label_smoothing": [0.0, 0.1]
    }

    best_val_acc = 0
    best_model = None
    best_device = None
    best_params = None

    for lr, dropout, ls in product(param_grid["lr"], param_grid["dropout"], param_grid["label_smoothing"]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BrailleCNN(dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        EPOCHS = 10
        for epoch in range(EPOCHS):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        val_acc = evaluate_model(val_loader, model, device, silent=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_device = device
            best_params = {"lr": lr, "dropout": dropout, "label_smoothing": ls}

    return best_model, best_device, best_params

def evaluate_model(loader, model, device, silent=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    if not silent:
        print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def predict_braille_text_from_image(img_path, model, device):
    img = cv.imread(str(img_path))
    thresh = preprocess_image(img)

    cv.imshow("Thresh", thresh)
    cv.waitKey(0)

    dot_contours = segment_braille_dots(thresh)
    cell_groups = group_dot_contours_into_cells(dot_contours)
    cells, _ = extract_cells(thresh, cell_groups)

    predicted_text = ''
    model.eval()

    for cell in cells:
        resized = cv.resize(cell, (28, 28)).astype(np.float32) / 255.0
        tensor = torch.tensor(resized).unsqueeze(0).unsqueeze(0).to(device)  # shape: (1, 1, 28, 28)

        with torch.no_grad():
            output = model(tensor)
            pred_class = torch.argmax(output, dim=1).item()
            predicted_text += chr(97 + pred_class)

    return predicted_text


def get_actual_text_from_yolo_labels(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        yolo_lines = f.readlines()

    labels = []
    for line in yolo_lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        labels.append((cls_id, x_center, y_center))



    actual_text = ''.join(chr(97 + cls_id) for cls_id, _, _ in labels)
    return actual_text


def compare_predicted_and_actual(predicted_text, actual_text):
    min_len = min(len(predicted_text), len(actual_text))
    matches = sum(predicted_text[i] == actual_text[i] for i in range(min_len))
    accuracy = matches / max(len(actual_text), 1) * 100  # avoid divide-by-zero
    print(f"\nActual Text:    {actual_text}")
    print(f"Predicted Text: {predicted_text}")
    print(f"Character-level Accuracy: {accuracy:.2f}%")
    return accuracy

def segment_braille_dots(thresh_img: np.ndarray) -> list:
    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    dot_contours = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if 10 < area < 100:
            dot_contours.append(cnt)
    return dot_contours

def group_dot_contours_into_cells(contours, eps=30, min_samples=1):
    centers = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        centers.append([x + w // 2, y + h // 2])

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    groups = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in groups:
            groups[label] = []
        groups[label].append(contours[idx])
    return list(groups.values())

def extract_cells(thresh_img: np.ndarray, cell_groups: list) -> tuple:
    cells, boxes = [], []

    for group in cell_groups:
        x1, y1, x2, y2 = np.inf, np.inf, -np.inf, -np.inf
        for cnt in group:
            x, y, w, h = cv.boundingRect(cnt)
            x1, y1 = min(x1, x), min(y1, y)
            x2, y2 = max(x2, x + w), max(y2, y + h)

        pad = 5
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(thresh_img.shape[1], x2 + pad), min(thresh_img.shape[0], y2 + pad)

        cell_img = thresh_img[y1:y2, x1:x2]
        cells.append(cell_img)
        boxes.append((x1, y1, x2, y2))
    return cells, boxes

if __name__ == "__main__":
    main()
