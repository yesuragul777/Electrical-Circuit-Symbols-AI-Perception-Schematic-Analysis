import os
import cv2
import numpy as np

def load_images_from_folder(folder_path, target_size=(64, 64)):
    images, labels = [], []
    classes = os.listdir(folder_path)
    for class_name in classes:
        class_path = os.path.join(folder_path, class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, target_size)
                images.append(img_resized)
                labels.append(class_name)
    return np.array(images), np.array(labels)

def normalize_images(images):
    return images / 255.0
