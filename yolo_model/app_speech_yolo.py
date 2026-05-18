import streamlit as st
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import io

# --- Load YOLO model for detection and classification ---
@st.cache_resource
def load_yolo_model():
    # Load your YOLO model. If it's trained for classification,
    # it will directly output class labels.
    # If it's a detection model that you want to use for classification,
    # make sure its output includes class predictions for each detected box.
    model = YOLO('braille_model_yolov11.pt')  # Your trained YOLOv8 model (detection + classification)
    return model

# --- Main app ---
def main():
    st.title("Braille Character Recognition.")

    # --- Load YOLO model ---
    yolo_model = load_yolo_model()
    
    # --- Define class mapping ---
    # This dictionary will map the integer output of your YOLO model to a character
    # You should update this to match the classes your YOLO model was trained on.
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
        st.write("Processing with YOLO model...")

        # --- Convert PIL image to OpenCV format ---
        image_cv = np.array(image_pil)
        image_cv_display = image_cv.copy() # Make a copy for drawing bounding boxes

        # --- Perform detection and classification with YOLO ---
        # The 'predict' method with a classification-trained YOLO model (or a detection model with class predictions)
        # will give us boxes and their corresponding class IDs.
        results = yolo_model.predict(source=image_cv, save=False, verbose=False)
        detections = results[0] # Assuming single image inference

        # --- Check if any detections were made ---
        if len(detections.boxes) == 0:
            st.warning("No Braille characters were detected in the image.")
            st.info("Please try another image or adjust the detection model's confidence threshold if necessary.")
        else:
            # --- Extract bounding box and class information for sorting ---
            detected_chars_info = []
            for i, box in enumerate(detections.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item()) # Get the class ID
                confidence = box.conf[0].item()    # Get the confidence

                detected_chars_info.append({
                    'box': (x1, y1, x2, y2),
                    'class_id': class_id,
                    'confidence': confidence
                })

            # --- Sort bounding boxes with y-tolerance ---
            # Define a y-tolerance. Adjust this value based on your images.
            Y_TOLERANCE = 50 # pixels - adjust as needed

            def sort_key(item):
                x1, y1, x2, y2 = item['box']
                return (y1 // Y_TOLERANCE, x1) # Sort by line (y-bin), then by x

            detected_chars_info.sort(key=sort_key)

            # --- Process each detection in sorted order ---
            results_text = ""
            for item in detected_chars_info:
                x1, y1, x2, y2 = item['box']
                class_id = item['class_id']
                confidence = item['confidence']
                
                # Get the actual character from the class map
                predicted_character = class_map.get(class_id, "Unknown")
                
                # --- Add to results string and draw bounding box ---
                results_text += predicted_character
                label = f"{predicted_character} ({confidence:.2f})"
                cv.rectangle(image_cv_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(image_cv_display, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            st.subheader("Processed Image")
            # --- Display the result image ---
            st.image(cv.cvtColor(image_cv_display, cv.COLOR_BGR2RGB), caption='Processed Image with Detections and Classifications.', use_container_width=True)
            
            st.subheader("Recognized Text")
            # --- Display the final recognized text ---
            st.success(f"**{results_text}**")

            # --- Add Text-to-Speech functionality ---
            st.subheader("Listen to the Recognized Text")
            if st.button("Generate Audio"):
                if results_text:
                    try:
                        tts = gTTS(text=results_text, lang='en')
                        audio_fp = io.BytesIO()
                        tts.write_to_fp(audio_fp)
                        st.audio(audio_fp, format='audio/mp3', start_time=0)
                    except Exception as e:
                        st.error(f"An error occurred during audio generation: {e}")
                else:
                    st.warning("No text to convert to speech.")


if __name__ == '__main__':
    main()