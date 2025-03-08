import os
import shutil
import yaml
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

def prepare_dataset(input_dir, output_dir='yolo_dataset', test_size=0.2):
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    class_ids = {cls: idx for idx, cls in enumerate(class_dirs)}
    
    # Create dataset structure
    splits = ['train', 'val']
    for split in splits:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, folder), exist_ok=True)

    # Process each class
    for cls in class_dirs:
        class_path = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        # Split images into train/val
        train_imgs, val_imgs = train_test_split(images, test_size=test_size, random_state=42)
        
        # Process splits
        for split, imgs in zip(splits, [train_imgs, val_imgs]):
            for img in imgs:
                # Copy image
                src = os.path.join(class_path, img)
                dst_img = os.path.join(output_dir, split, 'images', img)
                shutil.copy(src, dst_img)
                
                # Create label file (full-image bounding box)
                label_name = os.path.splitext(img)[0] + '.txt'
                dst_label = os.path.join(output_dir, split, 'labels', label_name)
                with open(dst_label, 'w') as f:
                    f.write(f"{int(class_ids[cls])} 0.5 0.5 1.0 1.0\n")

    # Create YAML config
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'names': {v: k for k, v in class_ids.items()}
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)
        
    return os.path.join(output_dir, 'dataset.yaml')

if __name__ == '__main__':
    # Configuration
    INPUT_DIR = 'dataset' #dataset path
    MODEL_TYPE = 'yolov8n.pt'
    EPOCHS = 10
    IMG_SIZE = 640
    BATCH = 16
    
    # Prepare dataset
    yaml_path = prepare_dataset(INPUT_DIR)
    
    # Initialize and train model
    model = YOLO(MODEL_TYPE)
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=0,
        plots=True,
        verbose=True,
        exist_ok=True,
        amp=False,
        workers=0,
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        optimizer='AdamW'
    )
    
    # Save final model
    model.save('yolov8_detection.pt')