import cv2
import joblib
import numpy as np
import argparse

def predict_symbol(image_path):
    model, encoder = joblib.load("models/symbol_svm_model.pkl")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (64, 64))
    img_normalized = img_resized / 255.0
    flat = img_normalized.flatten().reshape(1, -1)
    prediction = model.predict(flat)
    predicted_label = encoder.inverse_transform(prediction)[0]
    return predicted_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    result = predict_symbol(args.img_path)
    print(f"Predicted Symbol: {result}")
