import cv2 as cv
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical
from keras import models, layers

DATASET_DIR = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Dataset"

#Preprocessing the image
def preprocess_image(img : np.ndarray) -> np.ndarray:    
    _, thresh = cv.threshold(
        cv.Canny(
        cv.GaussianBlur(
        cv.cvtColor(
        cv.imread(img), 
        cv.COLOR_BGR2GRAY), 
        (1,1), 1),
        130, 255),
        130, 255, cv.THRESH_BINARY_INV)
    return thresh

# X represents numpy array for each pixel of image
# Y is label for each image
X, y = [], []
label_map = {chr(97 + i): i for i in range(26)}

for img in Path(DATASET_DIR).iterdir():
    preprocess = preprocess_image(img)
    X.append(preprocess)
    y.append(label_map[img.name[0]])

X = np.array(X)
y = np.array(y)

X = X.reshape(-1, 28, 28, 1) / 255.0  # Normalize 
y = to_categorical(y, 26) # Dimensions and number of classes

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN MODEL
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), #Img height and weight
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax') #Number of classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# TRAINING
model.fit(X_train, y_train, epochs=32, validation_data=(X_test, y_test))

# EVALUATION
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")