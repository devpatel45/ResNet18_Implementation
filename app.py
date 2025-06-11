import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

from model.resnet import ResNet18
from gradcam import GradCAM, overlay_heatmap

# Load your trained model
@st.cache_resource
def load_model():
    model = ResNet18(num_classes=10)
    model.load_state_dict(torch.load('best_resnet18.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

st.title("ResNet18 with Grad-CAM Visualization")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_prob = torch.softmax(output, dim=1)
        pred_class = pred_prob.argmax(dim=1).item()
        confidence = pred_prob[0, pred_class].item()

        classes = {'F_Breakage': 0, 'F_Crushed': 1, 'F_Normal': 2, 'R_Breakage': 3, 'R_Crushed': 4, 'R_Normal': 5}
        
        predicted_class_name = ""
        for k, v in classes.items():
            if v == pred_class:
                predicted_class_name = k

    st.write(f"**Prediction:** Class {predicted_class_name} with confidence {confidence:.2f}")

    if st.button("Show Grad-CAM Heatmap"):
    # Grad-CAM
        target_layer = model.layer4[-1].conv2
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(input_tensor, class_idx=pred_class)

        # Overlay heatmap
        img_np = np.array(image)
        overlay = overlay_heatmap(img_np, cam)
        


        st.image(overlay, caption="Grad-CAM Heatmap Overlay", use_column_width=True)
