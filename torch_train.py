import cv2 as cv
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

DATASET_DIR = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Dataset"

def main():
    X, y = [], []
    label_map = {chr(97 + i): i for i in range(26)} 

    for img_path in Path(DATASET_DIR).iterdir():
        processed = preprocess_image(cv.imread(img_path))
        X.append(processed)
        y.append(label_map[img_path.name[0]])

    X = np.array(X).astype(np.float32).reshape(-1, 1, 28, 28) / 255.0 
    y = np.array(y).astype(np.int64)

    train_loader, test_loader = split_into_loaders(X, y)
    model, device = train_model(train_loader)

    evaluate_model(test_loader, model, device)

def preprocess_image(img : np.ndarray) -> np.ndarray:    
    _, thresh = cv.threshold(
        cv.Canny(
        cv.GaussianBlur(
        cv.cvtColor(
        img, 
        cv.COLOR_BGR2GRAY), 
        (1,1), 1),
        130, 255),
        130, 255, cv.THRESH_BINARY_INV)
    return thresh

def split_into_loaders(X: np.array, y: np.array):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

class BrailleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3),  # input channels = 1
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),  # 28x28 -> conv & pool -> 5x5
            nn.ReLU(),
            nn.Linear(128, 26)  # 26 output classes
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

def train_model(train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrailleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 32
    for epoch in range(EPOCHS):
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

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/total:.4f} | Accuracy: {100 * correct / total:.2f}%")

    return model, device

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

    torch.save(model.state_dict(), "braille_cnn_model.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
