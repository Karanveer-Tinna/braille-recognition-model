import streamlit as st
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
# Ensure you have your CNN model definition in a file named model_cnn.py
from model_cnn import BrailleCNN 

# --- Load YOLO model for object detection ---
@st.cache_resource
def load_yolo_model():
    # Make sure 'braille_model_2.pt' is the correct path to your trained YOLO model
    model = YOLO('braille_model_2.pt') 
    return model

# --- Load Braille classification model ---
@st.cache_resource
def load_braille_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Adjust num_classes if your model was trained on a different number
    braille_classifier = BrailleCNN(num_classes=26)
    # Make sure 'yolo_classify_model_100.pt' is the correct path to your trained CNN weights
    braille_classifier.load_state_dict(torch.load("yolo_classify_model_100.pt", map_location=device))
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
    # These parameters for adaptiveThreshold might need tuning for your specific image conditions
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
    # Creates a mapping from class index (0-25) to letter ('A'-'Z')
    class_map = {i: chr(65 + i) for i in range(26)}

    # --- File uploader ---
    uploaded_file = st.file_uploader("Choose a Braille image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, caption='Uploaded Image.', use_container_width=True)
        st.write("")
        st.write("Processing...")

        # Convert PIL Image to OpenCV format
        image_cv = np.array(image_pil)
        image_cv = cv.cvtColor(image_cv, cv.COLOR_RGB2BGR)

        # --- Step 1: Perform detection with YOLO model ---
        detections = yolo_model.predict(source=image_cv, save=False, verbose=False)[0]

        if len(detections.boxes) == 0:
            st.warning("No Braille characters were detected in the image.")
        else:
            # --- Step 2: Store all detections before processing ---
            detected_characters = []
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped_img = image_cv[y1:y2, x1:x2]
                if cropped_img.size > 0:
                    detected_characters.append({'box': (x1, y1, x2, y2), 'image': cropped_img})

            # --- Step 3: Sort detections top-to-bottom, then left-to-right ---
            if detected_characters:
                # First, sort all characters by their top y-coordinate
                detected_characters.sort(key=lambda item: item['box'][1])

                # Then, group characters into lines based on y-coordinate with a tolerance
                lines = []
                current_line = []
                # Adjust this tolerance based on the line spacing in your images
                Y_TOLERANCE = 20 # in pixels

                if detected_characters:
                    current_line.append(detected_characters[0])
                    for char in detected_characters[1:]:
                        # Check if the current character is on the same line as the last one
                        # We compare the 'y' of the new char with the 'y' of the first char in the current line
                        if abs(char['box'][1] - current_line[0]['box'][1]) < Y_TOLERANCE:
                            current_line.append(char)
                        else:
                            # New line found. Sort the completed line by x-coordinate and add to lines
                            current_line.sort(key=lambda item: item['box'][0])
                            lines.append(current_line)
                            # Start a new line
                            current_line = [char]
                    
                    # Add the last line
                    if current_line:
                        current_line.sort(key=lambda item: item['box'][0])
                        lines.append(current_line)

                # Flatten the list of lines into a single list of characters in the correct order
                sorted_characters = [char for line in lines for char in line]
            else:
                sorted_characters = []


            # --- Step 4: Process the *sorted* detections ---
            results_text = ""
            for item in sorted_characters:
                x1, y1, x2, y2 = item['box']
                cropped = item['image']

                # --- Preprocess and classify ---
                processed = preprocess_for_cnn(cropped)
                try:
                    processed = transform(processed)
                except Exception as e:
                    st.error(f"Error transforming image: {e}")
                    continue

                # Ensure tensor is in the correct shape [C, H, W] -> [1, 1, 28, 28]
                if processed.shape != (1, 28, 28):
                    st.warning(f"Skipping a character due to unexpected processed shape: {processed.shape}")
                    continue
                
                input_tensor = processed.unsqueeze(0).to(device)
                with torch.no_grad():
                    output = braille_classifier(input_tensor)
                    class_confidence = torch.softmax(output, dim=1)[0]
                    predicted_class_index = class_confidence.argmax().item()
                    
                    predicted_character = class_map.get(predicted_class_index, "?")
                    confidence_score = class_confidence[predicted_class_index].item()

                # Add to results string and draw on the image
                results_text += predicted_character
                label = f"{predicted_character} ({confidence_score:.2f})"
                cv.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(image_cv, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            st.subheader("Processed Image")
            st.image(cv.cvtColor(image_cv, cv.COLOR_BGR2RGB), caption='Processed Image with Detections.', use_container_width=True)
            
            st.subheader("Recognized Text")
            st.success(f"**{results_text}**")

if __name__ == '__main__':
    main()