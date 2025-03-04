# Import required libraries
import streamlit as st  # Web app framework
import numpy as np  # Numerical operations
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
import torchvision.models as models  # Pretrained vision models
import torchvision.transforms as transforms  # Image transformations
import mlflow  # Experiment tracking
import mlflow.pytorch  # MLflow PyTorch integration
from PIL import Image  # Image processing
from torchvision.models import VGG16_Weights  # VGG16 pretrained weights


class DeepDreamModel(nn.Module):
    def __init__(self):
        """Initialize the Deep Dream model using VGG16 features"""
        super(DeepDreamModel, self).__init__()

        # Load pretrained VGG16 up to the 23rd layer (before fully connected layers)
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23]

        # Freeze all VGG parameters - we don't want to train the model, just use its features
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Forward pass through VGG16 feature extractor"""
        return self.vgg(x)  # Return feature maps from selected layers


# Initialize and configure the model
model = DeepDreamModel()  # Create model instance
model.eval()  # Set to evaluation mode (important for batchnorm/dropout layers)

# ----------------- Streamlit UI Setup -----------------
st.title("Deep Dream with VGG")  # Web app title
st.write("Upload an image and apply Deep Dream transformations.")  # Description

# Image upload widget (supports PNG, JPG, JPEG)
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

# Intensity slider for dream effect (1-10 scale, default 5)
intensity = st.slider("Dream Intensity", 1, 10, 5)


# ----------------- Image Processing Function -----------------
def deep_dream_model(image, intensity):
    """Apply Deep Dream transformation to input image
    Args:
        image: PIL Image - Input image to transform
        intensity: int - Strength of dream effect (1-10)
    Returns:
        PIL Image - Transformed Deep Dream image
    """
    # Define image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to VGG input size
        transforms.ToTensor(),  # Convert to [0,1] range tensor
        transforms.Normalize(  # ImageNet normalization
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Apply transformations and add batch dimension
    img_tensor = transform(image).unsqueeze(0)

    # ----------------- MLflow Tracking -----------------
    with mlflow.start_run():  # Start MLflow experiment tracking
        mlflow.log_param("dream_intensity", intensity)  # Log intensity parameter
        mlflow.pytorch.log_model(model, "deep_dream_model")  # Log model architecture

    # ----------------- Gradient Ascent Optimization -----------------
    img_tensor.requires_grad = True  # Enable gradient calculation for input image

    # Adam optimizer with learning rate scaled by intensity
    optimizer = torch.optim.Adam([img_tensor], lr=0.01 * intensity)

    # Feature enhancement loop
    for _ in range(20):  # Number of optimization steps
        optimizer.zero_grad()  # Reset gradients
        features = model(img_tensor)  # Get features from VGG
        loss = features.norm()  # Calculate loss as feature magnitude
        loss.backward()  # Backpropagate gradients
        optimizer.step()  # Update image tensor

    # ----------------- Post-processing -----------------
    # Reverse normalization: convert back to original pixel range
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    img_tensor = img_tensor * std + mean  # Denormalize

    # Ensure valid pixel values (0-1 range)
    img_tensor = torch.clamp(img_tensor, 0, 1)

    # Convert tensor to PIL Image
    img_np = img_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)  # Scale to 0-255 range
    return Image.fromarray(img_np)  # Convert numpy array to PIL Image


# ----------------- Image Processing Flow -----------------
if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Process image when button is clicked
    if st.button("Apply Deep Dream"):
        #st.write("Processing... Please wait.")
        transformed_image = deep_dream_model(image, intensity)
        st.image(transformed_image, caption="Deep Dream Image", use_column_width=True)