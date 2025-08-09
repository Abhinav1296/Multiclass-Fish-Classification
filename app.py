import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd

# ===============================
# App Config
# ===============================
st.set_page_config(page_title="🐟 Multiclass Fish Classifier", page_icon="🐟", layout="wide")

# Inject custom CSS for next-level UI polish
st.markdown("""
    <style>
    /* Hide Streamlit default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Background gradient */
    .stApp {
        background: linear-gradient(120deg, #e0f7fa, #f1f8ff);
        background-attachment: fixed;
        color: #034f84;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title styling */
    .title {
        text-align: center;
        padding: 0.5em 0;
        font-size: 2.3em;
        font-weight: bold;
        color: #012b4a;
    }

    /* Prediction box */
    .big-pred {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 4px 15px rgba(0,0,0,0.08);
        text-align: center;
        font-size: 1.3em;
        margin-top: 20px;
    }

    /* Confidence bar style */
    .stProgress > div > div {
        border-radius: 8px;
    }

    </style>
""", unsafe_allow_html=True)

# ===============================
# Model & Config
# ===============================
MODEL_PATH = 'resnet50_best.pth'
CLASS_NAMES = [
    'animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream', 'fish sea_food horse_mackerel',
    'fish sea_food red_mullet', 'fish sea_food red_sea_bream',
    'fish sea_food sea_bass', 'fish sea_food shrimp',
    'fish sea_food striped_red_mullet', 'fish sea_food trout'
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# ===============================
# Preprocessing
# ===============================
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# Header
# ===============================
st.markdown("<div class='title'>🐟 Multiclass Fish Image Classifier</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a fish image to identify its species with AI-powered precision.</p>", unsafe_allow_html=True)

# ===============================
# File Upload
# ===============================
col1, col2 = st.columns([1, 1])

with col1:
    image_file = st.file_uploader("📂 **Choose a fish image...**", type=["jpg", "jpeg", "png"])
    st.caption("Max size: 200MB · Supported formats: JPG, JPEG, PNG")

with col2:
    st.empty()

# ===============================
# Prediction
# ===============================
if image_file:
    image = Image.open(image_file).convert('RGB')

    col_prev, col_pred = st.columns([1, 1])
    with col_prev:
        st.image(image, caption="Uploaded Image", width=350)

    # Preprocess
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        top_idx = np.argmax(probabilities)
        top_prob = probabilities[top_idx]
        top_class = CLASS_NAMES[top_idx]

    with col_pred:
        st.markdown(f"<div class='big-pred'>Prediction:<br><strong>{top_class}</strong><br>Confidence: <strong>{top_prob*100:.2f}%</strong></div>", unsafe_allow_html=True)

    # Confidence Table + Visualization
    st.subheader("📊 Confidence Scores")
    prob_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Probability (%)': probabilities * 100
    }).sort_values(by="Probability (%)", ascending=True)

    st.bar_chart(data=prob_df.set_index('Class'))

    # Highlight top 3 predictions
    st.subheader("🏆 Top 3 Predictions")
    top3_idx = np.argsort(probabilities)[::-1][:3]
    for idx in top3_idx:
        st.write(f"**{CLASS_NAMES[idx]}** — {probabilities[idx]*100:.2f}%")
        st.progress(float(probabilities[idx]))

else:
    st.info("⬆️ Upload a fish image to start prediction.")

# ===============================
# Footer Branding
# ===============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em; color: #555;'>
        <strong>Model Used:</strong> ResNet50 | 
        <strong>Developed By:</strong> Abhinav Viswanathula ❤️<br>
    </div>
    """,
    unsafe_allow_html=True
)