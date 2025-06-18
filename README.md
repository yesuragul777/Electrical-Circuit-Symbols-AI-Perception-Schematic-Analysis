# 🧠 Electrical Circuit Symbol Recognition – AI Schematic Analyzer

This project uses computer vision and deep learning to identify and classify symbols in electrical schematics, such as resistors, capacitors, and diodes. The goal is to automate symbol recognition in engineering diagrams and reduce human error.

## 🔍 Features

- CNN-based model for symbol classification (Keras)
- SVM-based backup classifier with optimized parameters
- Preprocessing pipeline for grayscale + normalized input
- CLI for uploading and predicting new circuit symbols

## 🗂️ Dataset Structure

Place labeled images in `data/circuit_symbols/` like this:

circuit_symbols/

├── resistor/

├── capacitor/

├── inductor/

├── diode/

└── switch/


## 🧪 Train the Model

```bash
python src/train_cnn.py
python src/train_svm.py
```

## 📊 Evaluate

```bash
python src/evaluate.py
```

## ⚙️ Predict a New Symbol

```bash
python app/predict_symbol.py --img_path sample.png
```

## 📈 Results

- Accuracy: CNN: 96.8%, SVM: 94.2%

- See results in results/ folder.
