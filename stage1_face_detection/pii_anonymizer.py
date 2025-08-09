import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

class PIIAnonymizer:
    def __init__(self, model_path="pii_detection/face_license_detection_phase2/weights/best.pt"):
        """
        Initialize PII Anonymizer
        model_path: Path to your trained YOLO model
        """
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Please train the model first using PII_Detection.py")
            return
            
        self.model = YOLO(model_path)
        # Lower confidence thresholds for license plates
        self.confidence_thresholds = {
            'License_Plate': 0.25,  # Lower threshold for license plates
            'face': 0.5             # Higher threshold for faces
        }
        
    def blur_region(self, image, x1, y1, x2, y2, blur_strength=23):
        """Apply Gaussian blur to a specific region"""
        roi = image[y1:y2, x1:x2]
        if roi.size > 0:  # Check if ROI is valid
            blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 30)
            image[y1:y2, x1:x2] = blurred
        return image
    
    def pixelate_region(self, image, x1, y1, x2, y2, pixel_size=10):
        """Apply pixelation to a specific region"""
        roi = image[y1:y2, x1:x2]
        if roi.size > 0:
            h, w = roi.shape[:2]
            # Resize down
            small = cv2.resize(roi, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
            # Resize back up
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            image[y1:y2, x1:x2] = pixelated
        return image
    
    def anonymize_image(self, image_path, output_path=None, method='blur'):
        """
        Anonymize PII in an image
        method: 'blur', 'pixelate', or 'black'
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
            
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
            
        original_image = image.copy()
        
        # Run inference with lower confidence
        results = self.model(image_path, conf=0.1, verbose=False)
        
        pii_detected = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates and info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    # Use class-specific thresholds
                    threshold = self.confidence_thresholds.get(class_name, 0.5)
                    
                    if conf > threshold:
                        # Record detection
                        pii_detected.append({
                            'type': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        })
                        
                        # Apply anonymization based on method
                        if method == 'blur':
                            roi = image[y1:y2, x1:x2]
                            if roi.size > 0:
                                if class_name == 'License_Plate':
                                    # Stronger blur for license plates
                                    blurred = cv2.GaussianBlur(roi, (51, 51), 50)
                                else:  # face
                                    blurred = cv2.GaussianBlur(roi, (23, 23), 30)
                                image[y1:y2, x1:x2] = blurred
                        
                        elif method == 'black':
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        # Save anonymized image
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Anonymized image saved to: {output_path}")
        
        return {
            'anonymized_image': image,
            'original_image': original_image,
            'detections': pii_detected,
            'total_pii_found': len(pii_detected)
        }
    
    def process_folder(self, input_folder, output_folder, method='blur'):
        """Process all images in a folder"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        processed_count = 0
        total_pii_count = 0
        
        for image_file in input_path.iterdir():
            if image_file.suffix.lower() in image_extensions:
                output_file = output_path / f"anonymized_{image_file.name}"
                
                result = self.anonymize_image(
                    str(image_file), 
                    str(output_file), 
                    method=method
                )
                
                if result:
                    processed_count += 1
                    total_pii_count += result['total_pii_found']
                    print(f"Processed: {image_file.name} - Found {result['total_pii_found']} PII elements")
        
        print(f"\nProcessing complete!")
        print(f"Images processed: {processed_count}")
        print(f"Total PII elements anonymized: {total_pii_count}")

# Example usage
if __name__ == '__main__':
    # Initialize anonymizer (make sure your model is trained first)
    anonymizer = PIIAnonymizer()
    
    # Test on single image
    if os.path.exists("test_image3.jpg"):
        result = anonymizer.anonymize_image(
            "test_image3.jpg", 
            "anonymized_test3.jpg", 
            method='blur'
        )
        if result:
            print(f"Found {result['total_pii_found']} PII elements")
            
    # Process entire folder
    # anonymizer.process_folder("input_images", "anonymized_output", method='blur')