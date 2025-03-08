from ultralytics import YOLO
import cv2
import numpy as np

def classify_image(model_path, image_path):
    model = YOLO(model_path)
    results = model.predict(image_path, verbose=False)[0]
    
    if results.probs is None:
        print("Error: The model does not provide classification probabilities.")
        return None, None

    names = model.names
    probs = results.probs.data.tolist()
    top_class_index = np.argmax(probs)
    top_class_name = names[top_class_index]
    top_class_prob = probs[top_class_index]
    
    print(f"Image Classification Result:")
    print(f"Class: {top_class_name}")
    print(f"Confidence: {top_class_prob:.2f}")
    
    return top_class_name, top_class_prob

if __name__ == "__main__":
    model_path = "yolov8_classification.pt"
    image_path = "testing image path"

    classify_image(model_path, image_path)
