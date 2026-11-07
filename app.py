"""
Interactive Multi-Model Alzheimer's Disease Classification Web App.
Built with Streamlit and PyTorch.
"""

import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Alzheimer's MRI Classifier",
    page_icon="🧠",
    layout="wide"
)

# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
NUM_CLASSES = 4

# Check for weight paths (either in root or in weights/)
def find_weight_file(filename):
    candidates = [filename, os.path.join("weights", filename), os.path.join("..", filename)]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def build_model(model_name):
    """Initializes model architecture with 4 output classes."""
    if model_name == "ResNet18":
        base = models.resnet18(weights=None)
        base.fc = torch.nn.Linear(base.fc.in_features, NUM_CLASSES)
        weight_file = "resnet18.pth"
    elif model_name == "ResNet50":
        base = models.resnet50(weights=None)
        base.fc = torch.nn.Linear(base.fc.in_features, NUM_CLASSES)
        weight_file = "resnet50.pth"
    elif model_name == "VGG16":
        base = models.vgg16(weights=None)
        base.classifier[6] = torch.nn.Linear(base.classifier[6].in_features, NUM_CLASSES)
        weight_file = "vgg16.pth"
    elif model_name == "GoogLeNet":
        base = models.googlenet(weights=None)
        base.fc = torch.nn.Linear(base.fc.in_features, NUM_CLASSES)
        weight_file = "googleNet.pth"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    weight_path = find_weight_file(weight_file)
    return base, weight_path

# Image Preprocessing Transformation
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Application Header
st.title("🧠 Alzheimer's Disease MRI Classifier & Multi-Model Benchmark")
st.markdown("""
This application performs real-time classification of axial brain MRI scans across 4 stages of Alzheimer's Disease 
using deep convolutional neural networks fine-tuned on the OASIS cohort.
""")

col_sidebar = st.sidebar
col_sidebar.header("⚙️ Model Configuration")
selected_model_name = col_sidebar.selectbox(
    "Select Model Backbone:",
    ["ResNet18", "GoogLeNet", "ResNet50", "VGG16"],
    index=0
)

# Tabs
tab_infer, tab_metrics, tab_compare = st.tabs(["🔍 Live MRI Inference", "📊 Model Metrics & Curves", "📈 Benchmark Comparison"])

# --- TAB 1: INFERENCE ---
with tab_infer:
    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("1. Upload Brain MRI Scan")
        uploaded_file = st.file_uploader("Choose an axial MRI slice (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])

    with col_result:
        st.subheader("2. Diagnostic Inference")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded MRI Scan', width=260)

            # Load model
            base_model, weight_path = build_model(selected_model_name)
            if weight_path and os.path.exists(weight_path):
                base_model.to(device)
                try:
                    base_model.load_state_dict(torch.load(weight_path, map_location=device))
                    base_model.eval()

                    # Run inference
                    tensor_img = transformations(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = base_model(tensor_img)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                        pred_idx = np.argmax(probs)
                        pred_class = CLASSES[pred_idx]

                    st.success(f"**Predicted Diagnosis:** {pred_class} ({probs[pred_idx] * 100:.2f}% confidence)")

                    # Plot confidence bar chart
                    fig, ax = plt.subplots(figsize=(6, 3))
                    bars = ax.barh(CLASSES, probs * 100, color=['#4e79a7', '#f28e2b', '#59a14f', '#e15759'])
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Probability (%)")
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error executing model inference: {e}")
            else:
                st.warning(f"⚠️ Pretrained weights for **{selected_model_name}** were not found locally.")
                st.info("Download pre-trained weights from the GitHub Releases page and place them in the `weights/` directory.")
        else:
            st.info("Please upload an MRI image slice on the left to view inference results.")

# --- TAB 2: METRICS & CURVES ---
with tab_metrics:
    st.subheader(f"Training History & Validation Metrics: {selected_model_name}")

    metrics_data = {
        "ResNet18": {
            "loss": [1.0610, 0.8878, 0.8462, 0.8281, 0.8117],
            "acc": [59.52, 60.79, 63.56, 64.79, 64.87],
            "cm": np.array([[1909, 79, 200, 503], [184, 1664, 24, 62], [445, 29, 1888, 561], [792, 66, 636, 1152]])
        },
        "ResNet50": {
            "loss": [1.0854, 0.9012, 0.8354, 0.7981, 0.7654],
            "acc": [59.72, 61.54, 58.71, 60.22, 63.91],
            "cm": np.array([[1909, 109, 466, 242], [137, 1660, 81, 26], [413, 48, 2205, 225], [819, 83, 1033, 738]])
        },
        "VGG16": {
            "loss": [1.1912, 0.8736, 0.7939, 0.7334, 0.6939],
            "acc": [54.94, 62.38, 66.15, 67.24, 70.12],
            "cm": np.array([[1548, 79, 359, 666], [12, 1930, 3, 11], [186, 39, 2087, 549], [312, 73, 757, 1583]])
        },
        "GoogLeNet": {
            "loss": [1.0532, 0.8753, 0.8227, 0.7963, 0.7763],
            "acc": [62.43, 64.30, 64.52, 64.17, 64.63],
            "cm": np.array([[1547, 170, 588, 413], [40, 1816, 42, 22], [229, 63, 2278, 287], [507, 97, 1148, 947]])
        }
    }

    current_data = metrics_data[selected_model_name]
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Training Loss Progression**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, 6), current_data["loss"], 'ro-', label="Cross-Entropy Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{selected_model_name} Training Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

    with c2:
        st.markdown("**Validation Accuracy Progression**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, 6), current_data["acc"], 'bo-', label="Accuracy (%)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{selected_model_name} Accuracy")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

    st.markdown("**Confusion Matrix (Validation Cohort)**")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(current_data["cm"], cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=30, ha='right')
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(current_data["cm"][i, j]), ha="center", va="center",
                    color="white" if current_data["cm"][i, j] > current_data["cm"].max()/2 else "black")
    plt.tight_layout()
    st.pyplot(fig)

# --- TAB 3: BENCHMARK COMPARISON ---
with tab_compare:
    st.subheader("Cross-Architecture Performance Benchmark")

    st.markdown("""
    | Architecture | Final Epoch Accuracy | Trainable Parameters | Strengths |
    | :--- | :---: | :---: | :--- |
    | **ResNet-18** | **64.87%** | 8.4 M | Lightweight, rapid convergence, robust residual features |
    | **ResNet-50** | **63.91%** | 23.5 M | Deeper feature extraction |
    | **GoogLeNet** | **64.63%** | 5.6 M | Highly compact, multi-scale inception receptive fields |
    | **VGG-16** | **70.12%** | 134.3 M | Highest individual capacity; requires Git LFS for storage |
    """)

    models_accuracies = {
        "ResNet18": [59.5, 60.7, 63.5, 64.8, 64.9],
        "ResNet50": [59.7, 61.5, 58.7, 60.2, 63.9],
        "VGG16": [54.9, 62.4, 66.1, 67.2, 70.1],
        "GoogLeNet": [62.4, 64.3, 64.5, 64.2, 64.6]
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    for m_name, acc_vals in models_accuracies.items():
        ax.plot(range(1, 6), acc_vals, marker='o', label=m_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Multi-Model Validation Accuracy Trajectory")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
