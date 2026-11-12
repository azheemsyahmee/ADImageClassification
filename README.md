# Alzheimer's Disease MRI Classifier & Multi-Model Inference App

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive clinical inference web application built with **Streamlit** and **PyTorch** for multi-stage Alzheimer's Disease (AD) classification from structural brain MRI scans.

The application allows users to upload axial MRI slices, dynamically select between 4 fine-tuned deep learning architectures (**ResNet-18**, **GoogLeNet**, **ResNet-50**, and **VGG-16**), inspect diagnostic probability distributions, and explore empirical validation metrics and training trajectories side-by-side.

---

## Features

* **Multi-Architecture Model Serving:** Instantly switch between 4 convolutional backbones via a sidebar selector.
* **Real-Time Probabilistic Inference:** Outputs class probabilities, confidence percentages, and visual horizontal bar charts across the 4 clinical dementia stages:
  1. `NonDemented`
  2. `VeryMildDemented`
  3. `MildDemented`
  4. `ModerateDemented`
* **Interactive Diagnostics & Metrics:** Built-in tabs displaying confusion matrices, epoch-by-epoch cross-entropy loss trajectories, and validation accuracy curves.
* **Cross-Model Benchmark Tab:** Side-by-side performance comparison table and validation convergence trajectories.

---

## Model Architectures & Benchmarks

All backbones were fine-tuned using cross-entropy loss and Adam optimization over the OASIS brain MRI cohort:

| Backbone | Input Resolution | Architecture Type | Model Parameters | Checkpoint Weight | Primary Strength |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ResNet-18** | $224 \times 224$ | Residual CNN | ~11.2 M | `resnet18.pth` (44.7 MB) | Lightweight, rapid inference, balanced accuracy |
| **GoogLeNet** | $224 \times 224$ | Inception CNN | ~5.6 M | `googleNet.pth` (22.5 MB) | Highly compact multi-scale receptive fields |
| **ResNet-50** | $224 \times 224$ | Deep Residual CNN | ~23.5 M | `resnet50.pth` (94.3 MB) | Deep bottleneck feature extraction |
| **VGG-16** | $224 \times 224$ | Deep Sequential CNN | ~134.3 M | `vgg16.pth` (537.1 MB) | High model capacity |

---

## Repository Structure

```
alzheimer-classifier-streamlit/
├── .git/                           # Preserved Git repository history
├── .gitattributes                  # Git LFS configuration for binary weights
├── .gitignore                      # Python and weight management ignore rules
├── requirements.txt                # Pinned dependencies for local / cloud setup
├── README.md                       # Comprehensive application documentation
├── app.py                          # Modern production Streamlit application entry point
├── streamlitProject.py             # Original core inference script
├── notebooks/
│   ├── AdRes18.ipynb               # ResNet-18 training, loss curves & confusion matrix
│   ├── AdGoogleNet.ipynb           # GoogLeNet training & evaluation
│   ├── AdRes50.ipynb               # ResNet-50 training & evaluation
│   └── AdVgg.ipynb                 # VGG-16 training & evaluation
└── weights/
    ├── README.md                   # Download instructions for large weights
    ├── googleNet.pth               # Pretrained GoogLeNet weights (22.5 MB)
    └── resnet18.pth                # Pretrained ResNet-18 weights (44.7 MB)
```

---

## Installation & Local Execution

### 1. Clone the Repository
```bash
git clone https://github.com/azheemsyahmee/ADImageClassification.git
cd ADImageClassification
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## Model Training & Reproducibility

Each model's training pipeline is documented in its respective Jupyter Notebook under `notebooks/`:
* [AdRes18.ipynb](notebooks/AdRes18.ipynb) — ResNet-18 pipeline
* [AdGoogleNet.ipynb](notebooks/AdGoogleNet.ipynb) — GoogLeNet pipeline
* [AdRes50.ipynb](notebooks/AdRes50.ipynb) — ResNet-50 pipeline
* [AdVgg.ipynb](notebooks/AdVgg.ipynb) — VGG-16 pipeline

To train a model from scratch, open the desired notebook and set the path to your OASIS dataset root.

---

## Deployment to Streamlit Community Cloud

This repository is pre-configured for 1-click continuous deployment on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Fork or push this repository to GitHub.
2. Log into Streamlit Cloud, connect your GitHub account, and select this repository.
3. Set the Main file path to `app.py`.
4. Deploy! Streamlit Cloud automatically installs dependencies from `requirements.txt`.


---

## Dataset Provenance & Attribution

The underlying models were trained using brain MRI slices originating from the **Open Access Series of Imaging Studies (OASIS)** initiative (Marcus et al., 2007), accessed via the [Kaggle imagesoasis distribution](https://www.kaggle.com/datasets/ninadaithal/imagesoasis/data). Raw MRI scans are not redistributed within this repository. Users must obtain the dataset directly from Kaggle and observe the respective platform access terms.
