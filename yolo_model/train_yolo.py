import torch
from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using adevice: {device}")

    model = YOLO('yolov8n.pt')
    
    model.to(device)

    data_config_path = Path('./braille_data.yaml')

    if not data_config_path.exists():
        print(f"Error: Data configuration file not found at '{data_config_path}'")
        print("Please create the 'braille_data.yaml' file and ensure the paths are correct.")
        exit()

    print("Starting YOLOv8 training...")
    results = model.train(
        data=str(data_config_path),
        epochs=2, 
        imgsz=640,
        batch=16,
        name='braille_detection_run'
    )
    print("Training complete.")

    print("\nEvaluating model performance on the validation set...")
    metrics = model.val()  
    print("Evaluation metrics:", metrics)

    
    best_model = YOLO('./runs/detect/braille_detection_run/weights/best.pt')

    test_img_dir = Path(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\test\images")
    
    test_image_paths = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))

    if not test_image_paths:
        print(f"No test images found in {test_img_dir}")
    else:
        print(f"\nRunning inference on test images...")
        predict_results = best_model.predict(source=str(test_img_dir), save=True)
        print(f"Inference complete. Results saved in the latest 'runs/detect/predict' folder.")