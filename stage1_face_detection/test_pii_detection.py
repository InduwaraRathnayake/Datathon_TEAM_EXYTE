from ultralytics import YOLO
import os

def infer(image, model_path="pii_detection/face_license_detection_phase2/weights/best.pt"):
    """
    Initialize PII Anonymizer
    model_path: Path to your trained YOLO model
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using PII_Detection.py")
        return
        
    model = YOLO(model_path)
    # confidence_threshold = 0.5

    results = model(image)
    results[0].show()

infer("dataset/infer/test_image6.jpg")