# Smile Classification Deep Learning Model

A simple Keras-based Sequential Deep Neural Network designed to classify binary facial expressions (happy vs. sad) from binary/grayscale image datasets.

## Overview

This project loads black-and-white face images from a directory, processes and normalizes their pixel intensities, trains a multi-layer Perceptron (Dense Neural Network), and evaluates it on test data.

### Key Components:
- **Data Preloading:** Uses Pillow (`PIL`) to open images, convert them to 1-bit mode, extract raw pixel intensities, and assign categorical labels (`[1, 0]` for happy, `[0, 1]` for sad).
- **Normalization:** Scales pixel values from the $[0, 255]$ range down to $[0, 1]$.
- **Neural Network Architecture:** Built using Keras `Sequential` API:
  - Input layer / Dense layer with 1024 units ($\text{ReLU}$ activation)
  - Hidden Dense layer with 512 units ($\text{ReLU}$ activation)
  - Hidden Dense layer with 128 units ($\text{ReLU}$ activation)
  - Output Dense layer with 2 units ($\text{Softmax}$ activation)
- **Compilation & Training:** Uses the Adam optimizer (`learning_rate=0.005`), categorical crossentropy loss, and trains for 1000 epochs.

---

## Directory Structure

Ensure your workspace is organized as follows before running the script:
```text
├── smiles_dataset/
│   ├── training_set/
│   │   ├── happy_01.png
│   │   ├── sad_01.png
│   │   └── ...
│   └── test_set/
│       └── happy_test.png
└── script.py
```

---

## Dependencies

Make sure you have the following Python libraries installed:
```bash
pip install pillow numpy tensorflow keras
```

---

## Running the Code

Execute the script from your terminal:
```bash
python script.py
```

### Output Interpretation
During testing, the model will output a prediction array rounded to the nearest whole number:
- `[1., 0.]` $\rightarrow$ **Happy face**
- `[0., 1.]` $\rightarrow$ **Sad face**