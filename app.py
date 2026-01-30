import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image

from utils.preprocess import preprocess_image, extract_patches

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Image Tampering Detection", layout="centered")
st.title("Image Tampering Detection & Region Visualization")
st.write("Upload an image to analyze for possible tampering.")

# ---------- MODEL LOADING (PLACEHOLDER FOR DAY 3) ----------
@st.cache_resource
def load_model():
    """
    Load the trained patch-level classifier.
    NOTE: Replace with actual model loading on Day 3.
    """
    model = None  # placeholder
    return model

model = load_model()

# ---------- HELPERS ----------
def read_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    return image

def dummy_inference(image):
    """
    TEMPORARY inference for Day 2.
    Produces a fake heatmap to verify the UI pipeline.
    Replace with real patch inference + aggregation on Day 4.
    """
    h, w, _ = image.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    cv2.circle(heatmap, (w // 2, h // 2), min(h, w) // 6, 1.0, -1)
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    confidence = 0.75
    return heatmap, confidence

def overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap_norm = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# ---------- UI ----------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = read_image(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            # Preprocess (same as training)
            proc = preprocess_image(image)

            # --- TEMP: dummy inference (Day 2) ---
            heatmap, confidence = dummy_inference(proc)
            result_img = overlay_heatmap(image, heatmap)

        st.subheader("Result")
        st.write(f"**Tampering confidence:** {confidence:.2f}")

        st.image(result_img, caption="Highlighted suspicious regions", use_column_width=True)

        st.info(
            "Note: This visualization is a placeholder. "
            "Real patch-level inference and localization will replace it."
        )

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Deep Learning–Based Image Tampering Detection and Region Visualization System")
