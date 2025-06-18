from preprocess import load_images_from_folder, normalize_images
from utils import encode_labels
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

images, labels = load_images_from_folder("data/circuit_symbols")
images = normalize_images(images)
flat_images = images.reshape(len(images), -1)
labels, encoder = encode_labels(labels)

svm = SVC(kernel='rbf', C=10)
svm.fit(flat_images, labels)
joblib.dump((svm, encoder), "models/symbol_svm_model.pkl")
