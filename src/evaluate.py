from sklearn.metrics import classification_report, confusion_matrix
from utils import plot_confusion_matrix
from preprocess import load_images_from_folder, normalize_images
from utils import encode_labels
import joblib
import numpy as np
import pandas as pd

images, labels = load_images_from_folder("data/circuit_symbols")
images = normalize_images(images)
flat_images = images.reshape(len(images), -1)
labels_encoded, encoder = encode_labels(labels)

svm, encoder_svm = joblib.load("models/symbol_svm_model.pkl")
y_pred = svm.predict(flat_images)

cm = confusion_matrix(labels_encoded, y_pred)
report = classification_report(labels_encoded, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv("results/performance_report.csv")

plot_confusion_matrix(cm, encoder.classes_, "SVM Confusion Matrix", "results/confusion_matrix_svm.png")
