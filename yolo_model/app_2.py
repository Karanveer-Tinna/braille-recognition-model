import streamlit as st
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
from model_cnn import BrailleCNN

# --- Load YOLO model for object detection ---
@st.cache_resource
def load_yolo_model():
    model = YOLO('braille_model_2.pt')  # your trained YOLOv8 detection model
    return model

# --- Load Braille classification model ---
@st.cache_resource
def load_braille_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    braille_classifier = BrailleCNN(num_classes=26)  # adjust if your CNN uses 26 classes
    braille_classifier.load_state_dict(torch.load("yolo_classify_model_100.pt", map_location=device))  # path to your trained CNN
    braille_classifier.to(device)
    braille_classifier.eval()
    return braille_classifier, device

# --- Transform for CNN ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# --- Preprocessing for CNN (same as training) ---
def preprocess_for_cnn(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    denoised = cv.bilateralFilter(equalized, 9, 100, 100)
    thresh = cv.adaptiveThreshold(denoised, 255,
                                  cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY_INV,
                                  blockSize=5, C=2)
    return thresh

# --- Main app ---
def main():
    st.title("Braille Character Recognition")

    # --- Load models ---
    yolo_model = load_yolo_model()
    braille_classifier, device = load_braille_classifier()
    
    # --- Define class mapping ---
    # This dictionary will map the integer output of your model to a character
    # You should update this to match the classes your model was trained on.
    # The order must be the same as the one used during training.
    # Example for A-Z:
    class_map = {i: chr(65 + i) for i in range(26)} # Creates {0: 'A', 1: 'B', ...}


    # --- File uploader ---
    uploaded_file = st.file_uploader("Choose a Braille image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # --- Display uploaded image ---
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, caption='Uploaded Image.', use_container_width=True)
        st.write("")
        st.write("Processing...")

        # --- Convert PIL image to OpenCV format ---
        image_cv = np.array(image_pil)
        image_cv = cv.cvtColor(image_cv, cv.COLOR_RGB2BGR)

        # --- Perform detection ---
        detections = yolo_model.predict(source=image_cv, save=False, verbose=False)[0]
        
        # --- Check if any detections were made ---
        if len(detections.boxes) == 0:
            st.warning("No Braille characters were detected in the image.")
            st.info("Please try another image or adjust the detection model's confidence threshold if necessary.")
        else:
            # --- Process each detection ---
            results_text = ""
            for i, box in enumerate(detections.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # --- Crop the detected character ---
                cropped = image_cv[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                
                # --- Preprocess and classify ---
                processed = preprocess_for_cnn(cropped)
                try:
                    processed = transform(processed)
                except:
                    continue

                if processed.shape != (1, 28, 28):
                    continue
                
                input_tensor = processed.unsqueeze(0).to(device)
                with torch.no_grad():
                    output = braille_classifier(input_tensor)
                    predicted_class_index = output.argmax(dim=1).item()
                    
                    # Get the actual character from the class map
                    predicted_character = class_map.get(predicted_class_index, "Unknown")
                    
                    class_confidence = torch.softmax(output, dim=1)[0][predicted_class_index].item()

                # --- Add to results string and draw bounding box ---
                results_text += predicted_character
                label = f"{predicted_character} ({class_confidence:.2f})"
                cv.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(image_cv, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            st.subheader("Processed Image")
            # --- Display the result image ---
            st.image(cv.cvtColor(image_cv, cv.COLOR_BGR2RGB), caption='Processed Image with Detections.', use_container_width=True)
            
            st.subheader("Recognized Text")
            # --- Display the final recognized text ---
            st.success(f"**{results_text}**")


if __name__ == '__main__':
    main()