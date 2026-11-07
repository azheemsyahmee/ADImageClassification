import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Available models
model_choices = {
    "ResNet18": ("resnet18.pth", models.resnet18(pretrained=True)),
    "ResNet50": ("resnet50.pth", models.resnet50(pretrained=True)),
    "VGG16": ("vgg16.pth", models.vgg16(pretrained=True)),
    "GoogLeNet": ("googleNet.pth", models.googlenet(pretrained=True)),
}

# Model selection
st.title("Image Classification App with Model Comparison")
st.write("Select a model to classify images and compare performance.")
selected_model_name = st.selectbox("Choose a model", list(model_choices.keys()))

# Load selected model
model_weights, base_model = model_choices[selected_model_name]

# Modify final layer for 4 classes
num_classes = 4
if selected_model_name == "GoogLeNet":
    base_model.fc = torch.nn.Linear(base_model.fc.in_features, num_classes)
elif selected_model_name == "VGG16":
    base_model.classifier[6] = torch.nn.Linear(base_model.classifier[6].in_features, num_classes)
else:
    base_model.fc = torch.nn.Linear(base_model.fc.in_features, num_classes)
    
model = base_model.to(device)

# Load model weights
model.load_state_dict(torch.load(model_weights, map_location=device))
model.eval()

# Define image transformations
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Class labels
classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Image upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    transformed = transformations(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_output = model(transformed)
        _, predicted = torch.max(predicted_output, 1)
        predicted_class = classes[predicted.item()]

    # Display prediction
    st.write(f"### Predicted Class: {predicted_class}")

# Display metrics and graphs for the selected model
if st.checkbox("Show Model Metrics and Graphs"):
    if selected_model_name == "ResNet18":
        # ResNet18 specific dummy metrics (replace with actual values)
        st.write("### Confusion Matrix")
        cf_matrix = np.array([[1909, 79, 200, 503], [184, 1664, 24, 62], [445, 29, 1888, 561], [792, 66, 636, 1152]])
        st.write(cf_matrix)

        st.write("### Classification Report")
        st.text(f"Model: {selected_model_name}\n")
        st.text("Precision, recall, F1-score, etc.")

        # Data provided
        epoch_nums = [0, 1, 2, 3, 4]
        training_loss = [1.06100637818216, 0.8877511064531982, 0.8462250353267757, 0.8281480188491523, 0.8117428503690227]

        # Streamlit app
        st.title("Training Loss Over Epochs")

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_nums, training_loss, marker='o', label='Training Loss', color='red')

        # Add labels, title, and legend
        plt.xlabel('Epoch Numbers')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)

        # Data provided
        epoch_nums = [0, 1, 2, 3, 4]
        training_acc = [59.52521090837748, 60.79066117323916, 63.556994310378656, 64.79301549931333, 64.8714930351187]

        # Streamlit app
        st.title("Training Accuracy Over Epochs")

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_nums, training_acc, marker='o', label='Training Accuracy')

        # Add labels, title, and legend
        plt.xlabel('Epoch Numbers')
        plt.ylabel('Training Accuracy (%)')
        plt.title('Training Accuracy Over Epochs')
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)

    elif selected_model_name == "ResNet50":
        # ResNet50 specific dummy metrics (replace with actual values)
        st.write("### Confusion Matrix")
        cf_matrix = np.array([[1909, 109, 466, 242], [137, 1660, 81, 26], [413, 48, 2205, 225], [819, 83, 1033, 738]])
        st.write(cf_matrix)

        st.write("### Classification Report")
        st.text(f"Model: {selected_model_name}\n")
        st.text("Precision, recall, F1-score, etc.")

        # Data provided
        epoch_nums = [0, 1, 2, 3, 4]
        training_loss = [1.0590778203901423, 0.9256694978603752, 0.8891262424889431, 0.8518411789930636, 0.8296267590535584]

        # Streamlit app
        st.title("Training Loss Over Epochs")

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_nums, training_loss, marker='o', label='Training Loss', color='red')

        # Add labels, title, and legend
        plt.xlabel('Epoch Numbers')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)

        # Data provided
        epoch_nums = [0, 1, 2, 3, 4]
        training_acc = [59.721404747890915, 61.5460074553659, 58.740435550323724, 60.1824602707475, 63.88071414557584]

        # Streamlit app
        st.title("Training Accuracy Over Epochs")

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_nums, training_acc, marker='o', label='Training Accuracy')

        # Add labels, title, and legend
        plt.xlabel('Epoch Numbers')
        plt.ylabel('Training Accuracy (%)')
        plt.title('Training Accuracy Over Epochs')
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)

    elif selected_model_name == "VGG16":
        # VGG16 specific dummy metrics (replace with actual values)
        st.write("### Confusion Matrix")
        cf_matrix = np.array([[1548, 79, 359, 666], [12, 1930, 3, 11], [186, 39, 2087, 549], [312, 73, 757, 1583]])
        st.write(cf_matrix)

        st.write("### Classification Report")
        st.text(f"Model: {selected_model_name}\n")
        st.text("Precision, recall, F1-score, etc.")

        # Data provided
        epoch_nums = [0, 1, 2, 3, 4]
        training_loss = [1.191193744059532, 0.8736235570683274, 0.7938908200751069, 0.7334304776803781, 0.6938790664557488]

        # Streamlit app
        st.title("Training Loss Over Epochs")

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_nums, training_loss, marker='o', label='Training Loss', color='red')

        # Add labels, title, and legend
        plt.xlabel('Epoch Numbers')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)

        # Data provided
        epoch_nums = [0, 1, 2, 3, 4]
        training_acc = [54.944084755738665, 62.37983127329802, 66.14675299195605, 67.23562880125564, 70.1196782421032]

        # Streamlit app
        st.title("Training Accuracy Over Epochs")

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_nums, training_acc, marker='o', label='Training Accuracy')

        # Add labels, title, and legend
        plt.xlabel('Epoch Numbers')
        plt.ylabel('Training Accuracy (%)')
        plt.title('Training Accuracy Over Epochs')
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)

    elif selected_model_name == "GoogLeNet":
        # GoogLeNet specific dummy metrics (replace with actual values)
        st.write("### Confusion Matrix")
        cf_matrix = np.array([[1547, 170, 588, 413], [40, 1816, 42, 22], [229, 63, 2278, 287], [507, 97, 1148, 947]])
        st.write(cf_matrix)

        st.write("### Classification Report")
        st.text(f"Model: {selected_model_name}\n")
        st.text("Precision, recall, F1-score, etc.")

        # Data provided
        epoch_nums = [0, 1, 2, 3, 4]
        training_loss = [1.053200613026337, 0.8752990481193348, 0.8227233103206081, 0.7963139312241667, 0.7762749329690011]

        # Streamlit app
        st.title("Training Loss Over Epochs")

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_nums, training_loss, marker='o', label='Training Loss', color='red')

        # Add labels, title, and legend
        plt.xlabel('Epoch Numbers')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)

        # Data provided
        epoch_nums = [0, 1, 2, 3, 4]
        training_acc = [62.42887973317638, 64.30253090052972, 64.51834412399451, 64.16519521287032, 64.6262507357269]

        # Streamlit app
        st.title("Training Accuracy Over Epochs")

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_nums, training_acc, marker='o', label='Training Accuracy')

        # Add labels, title, and legend
        plt.xlabel('Epoch Numbers')
        plt.ylabel('Training Accuracy (%)')
        plt.title('Training Accuracy Over Epochs')
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)

# Model comparison
if st.checkbox("Compare Models"):
    st.write("### Accuracy Comparison")

    # Model accuracies for comparison (replace with actual values)
    models_accuracies = {
        "ResNet18": [59.5, 60.7, 63.5, 64.8, 64.9],  # Replace with actual accuracy values
        "ResNet50": [59.7, 61.5, 58.7, 60.2, 63.9],  # Replace with actual accuracy values
        "VGG16": [54.9, 62.4, 66.1, 67.2, 70.1],     # Replace with actual accuracy values
        "GoogLeNet": [62.4, 64.3, 64.5, 64.2, 64.6]  # Replace with actual accuracy values
    }

    # Plot accuracy comparison
    for model_name, acc in models_accuracies.items():
        plt.plot(range(1, len(acc) + 1), acc, marker='o', label=model_name)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.legend()
    st.pyplot(plt)
