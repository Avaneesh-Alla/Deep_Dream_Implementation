# 🌀 Deep Dream Generator with PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-%230099cc.svg)](https://mlflow.org/)

A PyTorch implementation of Google's Deep Dream algorithm using VGG-16 for psychedelic image generation, featuring an interactive web interface.

![Deep Dream Example](images/dream_example.jpg) *(Replace with your example image)*

## 🧠 Key Concepts

##### 🌌 What is Deep Dream?
Deep Dream is a computer vision technique that amplifies patterns learned by neural networks, creating surreal, dream-like images by maximizing activation gradients.

![Deep Dream Process](images/dream_process.png) *(Add gradient ascent image)*

##### 🔍 Attention Models (Context)
While this project uses CNNs, attention models (like those in Transformers) focus on relevant image regions. Though not used here, they share the philosophy of emphasizing important features.

##### 🏛️ VGG-16 Architecture
The 16-layer CNN from Oxford's Visual Geometry Group, known for its simplicity and effectiveness in feature extraction.

```plaintext
VGG-16 Architecture:
Input -> 2x64 conv -> MaxPool -> 2x128 conv -> MaxPool -> 
3x256 conv -> MaxPool -> 3x512 conv -> MaxPool -> 3x512 conv -> MaxPool -> FC Layers
```

##### ⬆️ Gradient Ascent
Unlike gradient descent for loss minimization, we maximize layer activations through:

```python
for _ in range(steps):
    optimizer.zero_grad()
    loss = features.norm()  # Maximize activation
    loss.backward()
    optimizer.step()
```

## 🛠️ Project Structure

```plaintext
deep-dream-pytorch/
├── app.py               # Streamlit web interface
├── requirements.txt     # Dependencies
├── images/              # Sample images and diagrams
├── README.md            # This documentation
└── mlruns/              # MLflow experiment tracking
```

## ⚙️ Installation

##### Clone repository:
```bash
git clone https://github.com/yourusername/deep-dream-pytorch.git
cd deep-dream-pytorch
```

##### Install dependencies:
```bash
pip install -r requirements.txt
```

##### Run Streamlit app:
```bash
streamlit run main.py
```
