import os
import shutil
import yaml
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

def prepare_dataset(input_dir, output_dir='yolo_dataset', test_size=0.2):
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    splits = ['train', 'val']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    for cls in class_dirs:
        class_path = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        train_imgs, val_imgs = train_test_split(images, test_size=test_size, random_state=42)
        for split, imgs in zip(splits, [train_imgs, val_imgs]):
            split_class_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_class_dir, exist_ok=True)
            for img in imgs:
                shutil.copy(os.path.join(class_path, img), os.path.join(split_class_dir, img))
    return output_dir

if __name__ == '__main__':
    INPUT_DIR = 'Easy Yolo8/train' #training folder
    MODEL_TYPE = 'yolov8n-cls.pt'
    EPOCHS = 10
    IMG_SIZE = 224
    BATCH = 16

    dataset_path = prepare_dataset(INPUT_DIR)

    model = YOLO(MODEL_TYPE)
    results = model.train(
        data=dataset_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=0,
        plots=True,
        verbose=True
    )

    model.save('yolov8_classification.pt')