import cv2 as cv
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


DATASET_DIR = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Dataset"


# ----------- Main Pipeline -----------
def main():
    X, y = [], []
    label_map = {chr(97 + i): i for i in range(26)}

    for img_path in Path(DATASET_DIR).iterdir():
        processed = preprocess_image(cv.imread(str(img_path)))
        X.append(processed)
        y.append(label_map[img_path.name[0]])

    X = np.array(X).astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
    y = np.array(y).astype(np.int64)

    train_loader, test_loader = split_into_loaders(X, y)
    model, device = train_model(train_loader)
    evaluate_model(test_loader, model, device)

    # Test using YOLO labels (no segmentation)
    image_path = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\images\test\IMG_4479_jpg.rf.74bfb6d141e15e92aa9aa0cb70236cd5.jpg"
    label_path = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\labels\test\IMG_4479_jpg.rf.74bfb6d141e15e92aa9aa0cb70236cd5.txt"

    predicted = predict_braille_from_yolo(image_path, label_path, model, device)
    actual = get_actual_text_from_yolo_labels(label_path)

    compare_predicted_and_actual(predicted, actual)


# ----------- Preprocessing -----------
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


# ----------- Data Splitting -----------
def split_into_loaders(X: np.array, y: np.array):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


# ----------- CNN Model -----------
class BrailleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 26)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x


# ----------- Training -----------
def train_model(train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrailleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(32):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/32 | Loss: {running_loss/total:.4f} | Accuracy: {100 * correct / total:.2f}%")

    return model, device


# ----------- Evaluation -----------
def evaluate_model(test_loader, model, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    # torch.save(model.state_dict(), "braille_cnn_model.pth")


# ----------- YOLO-Powered Inference -----------
def predict_braille_from_yolo(img_path, label_path, model, device):
    image = cv.imread(img_path)
    height, width = image.shape[:2]

    boxes = []
    with open(label_path, 'r') as file:
        for line in file:
            cls_id, x_center, y_center, w, h = map(float, line.strip().split())
            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)
            boxes.append((x1, y1, x2, y2))

    # Sort boxes top-to-bottom, left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    predicted_text = ''
    model.eval()

    for x1, y1, x2, y2 in boxes:
        cell = image[y1:y2, x1:x2]
        gray = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, (28, 28)).astype(np.float32) / 255.0
        tensor = torch.tensor(resized).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred_class = torch.argmax(output, dim=1).item()
            predicted_text += chr(97 + pred_class)

    return predicted_text


# ----------- Load YOLO Ground Truth -----------
def get_actual_text_from_yolo_labels(label_path):
    with open(label_path, "r") as f:
        yolo_lines = f.readlines()

    labels = []
    for line in yolo_lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        labels.append(chr(97 + cls_id))

    return ''.join(labels)


# ----------- Compare Predicted vs Actual -----------
def compare_predicted_and_actual(predicted_text, actual_text):
    min_len = min(len(predicted_text), len(actual_text))
    matches = sum(predicted_text[i] == actual_text[i] for i in range(min_len))
    accuracy = matches / max(len(actual_text), 1) * 100
    print(f"\nActual Text:    {actual_text}")
    print(f"Predicted Text: {predicted_text}")
    print(f"Character-level Accuracy: {accuracy:.2f}%")
    return accuracy


# ----------- Entry Point -----------
if __name__ == "__main__":
    main()
