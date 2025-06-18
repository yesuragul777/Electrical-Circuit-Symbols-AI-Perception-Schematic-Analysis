from preprocess import load_images_from_folder, normalize_images
from utils import encode_labels
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

images, labels = load_images_from_folder("data/circuit_symbols")
images = normalize_images(images)
images = images.reshape(-1, 64, 64, 1)
labels, encoder = encode_labels(labels)
y = to_categorical(labels)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(set(labels)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(images, y, epochs=10, validation_split=0.2)

model.save("models/symbol_cnn_model.h5")
