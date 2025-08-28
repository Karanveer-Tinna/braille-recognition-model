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
import copy

# All class and function definitions are safe to keep outside the main guard
# --- OPTIMIZED CNN Model with Batch Norm and Dropout ---
class BrailleCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(BrailleCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_block = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc_block(x)
        return x

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
        if image is None: return None
        h, w = image.shape[:2]

        label_path = self.label_dir / (image_path.stem + ".txt")
        if not label_path.exists(): return None

        Xs, Ys = [], []
        with open(label_path) as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 5: continue
                class_id, x_c, y_c, box_w, box_h = map(float, values)
                xmin = int((x_c - box_w / 2) * w)
                ymin = int((y_c - box_h / 2) * h)
                xmax = int((x_c + box_w / 2) * w)
                ymax = int((y_c + box_h / 2) * h)
                if not (0 <= xmin < xmax <= w and 0 <= ymin < ymax <= h): continue
                cropped = image[ymin:ymax, xmin:xmax]
                if cropped.size == 0: continue
                processed = preprocess_image(cropped)
                if self.transform:
                    processed = self.transform(processed)
                if processed.shape != (1, 28, 28): continue
                Xs.append(processed)
                Ys.append(int(class_id))
        if not Xs: return None
        return torch.stack(Xs), torch.tensor(Ys)

# --- Collate function ---
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return torch.empty(0), torch.empty(0, dtype=torch.long)
    X, Y = zip(*batch)
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    return X, Y

# --- Training and Evaluation Functions ---
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for X, Y in loader:
        if X.numel() == 0: continue
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        _, predicted = outputs.max(1)
        total_samples += Y.size(0)
        total_correct += predicted.eq(Y).sum().item()
    if total_samples == 0: return 0, 0
    return total_loss / total_samples, 100 * total_correct / total_samples

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for X, Y in loader:
            if X.numel() == 0: continue
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            loss = criterion(outputs, Y)
            total_loss += loss.item() * X.size(0)
            _, predicted = outputs.max(1)
            total_samples += Y.size(0)
            total_correct += predicted.eq(Y).sum().item()
    if total_samples == 0: return 0, 0
    return total_loss / total_samples, 100 * total_correct / total_samples


# ====================================================================
# SCRIPT EXECUTION GUARD
# ====================================================================
if __name__ == '__main__':
    # --- Directories ---
    train_img_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images"
    train_lbl_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\labels"
    valid_img_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\valid\images"
    valid_lbl_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\valid\labels"
    test_img_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\test\images"
    test_lbl_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\test\labels"

    # --- Data Augmentation and Transforms ---
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    # --- Load Datasets ---
    train_dataset = YOLOBrailleDataset(train_img_dir, train_lbl_dir, train_transform)
    valid_dataset = YOLOBrailleDataset(valid_img_dir, valid_lbl_dir, val_test_transform)
    test_dataset  = YOLOBrailleDataset(test_img_dir, test_lbl_dir, val_test_transform)
    
    if len(train_dataset) == 0:
        print(f"Error: No images found in training directory: '{train_img_dir}'")
        sys.exit(1)

    # --- Create DataLoaders ---
    # NOTE: Set num_workers=0 if you continue to have issues, but the __main__ guard should fix it.
    # Set pin_memory to False since you are not on a CUDA device.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    loader_args = {'num_workers': 4, 'pin_memory': use_cuda} if use_cuda else {'num_workers': 0}
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, **loader_args)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, **loader_args)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, collate_fn=collate_fn, **loader_args)

    # --- Model, Loss, and Optimizer ---
    model = BrailleCNN(num_classes=26).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # --- Run Training with Early Stopping ---
    epochs = 100
    best_val_acc = 0
    patience, patience_counter = 15, 0 
    best_model_weights = None

    print("Starting training...")
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        
        print(f"Epoch {epoch+1:02d}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
              f"Valid Loss={val_loss:.4f}, Valid Acc={val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"  -> New best validation accuracy: {best_val_acc:.2f}%. Saving model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Validation accuracy did not improve for {patience} epochs. Stopping early.")
                break

    # --- Final Test ---
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    else:
        print("Warning: Early stopping was not triggered. Using the model from the last epoch.")
        
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")

    torch.save(model.state_dict(), "braille_classifier_best_model.pt")
    print("Best model saved successfully as 'braille_classifier_best_model.pt'")