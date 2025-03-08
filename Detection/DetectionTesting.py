import cv2
import torch
from ultralytics import YOLO

def detect_objects(model_path, image_path, conf_threshold=0.2):
    # Load the trained YOLOv8 model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(image_path, conf=conf_threshold, verbose=True)
    
    # Read image for visualization
    img = cv2.imread(image_path)

    # Get class names from the model
    names = model.names

    # Process detection results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = f"{names[cls]}: {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show image
    cv2.imshow("YOLOv8 Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = "yolov8_detection.pt"  # Replace with your model
    IMAGE_PATH = "testing image"  # Replace with your image path

    detect_objects(MODEL_PATH, IMAGE_PATH)
