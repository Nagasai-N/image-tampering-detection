import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image

from utils.inference import load_model, generate_heatmap
from utils.visualize import overlay_heatmap, draw_bounding_boxes
from styles_injector import inject_styles          # ← design loader

# ───────────────────────────────────────────────
#                PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Image Tampering Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_styles()                                    # ← injects styles.css

# ───────────────────────────────────────────────
#                     HEADER
# ───────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
  <div class="main-title">Image <span>Tampering</span> Detector</div>
  <div class="subtitle">Upload an image and let the model surface manipulated regions with heatmap localisation.</div>
  <div class="tag-row">
    <span class="tag">Deep Learning</span>
    <span class="tag">Heatmap Localisation</span>
    <span class="tag">Real-time</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
#                   SIDEBAR
# ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔎 Detection Scope")
    st.markdown("""
    - Copy-move forgery
    - Splicing attacks
    - Region removal / addition
    - Inpainting & generative fills
    - Other common manipulations
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ System Info")
    device_label = "GPU  ✓" if torch.cuda.is_available() else "CPU"
    st.caption(f"Device  ·  {device_label}")
    st.caption("Model  ·  PyTorch")
    st.caption("Interface  ·  Streamlit")

# ───────────────────────────────────────────────
#                DEVICE & MODEL
# ───────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading detection model…")
def get_model():
    return load_model("models/tamper_classifier.pth", device)


model = get_model()


# ───────────────────────────────────────────────
#                CLASSIFY FUNCTION
# ───────────────────────────────────────────────
def classify_image(heatmap):
    flat = heatmap.flatten()
    mu        = float(np.mean(flat))
    sigma     = float(np.std(flat)) + 1e-6
    h, w      = heatmap.shape
    image_area = h * w

    mask = heatmap > (mu + 1.5 * sigma)
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

    if num_labels <= 1:
        return "Original", 0.0, 0.0, heatmap

    largest_area = 0
    best_region  = None
    for lid in range(1, num_labels):
        region = (labels == lid)
        area   = np.sum(region)
        if area > largest_area:
            largest_area = area
            best_region  = region

    area_ratio  = largest_area / image_area
    region_mean = float(np.mean(heatmap[best_region]))

    is_tampered = (
        area_ratio   >= 0.015                    and
        region_mean  >= (mu + 2.0 * sigma)       and
        (region_mean / (mu + 1e-6)) >= 1.8
    )

    return ("Tampered" if is_tampered else "Original"), region_mean, area_ratio, heatmap


# ───────────────────────────────────────────────
#                UPLOAD CARD
# ───────────────────────────────────────────────
st.markdown("""
<div class="card">
  <div class="card-header">
    <div class="card-icon">📁</div>
    <div>
      <div class="card-title">Upload Image</div>
      <div class="card-desc">JPG · JPEG · PNG  —  recommended max ~10 MB</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & drop your image here, or click to browse",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG. Recommended max ~10 MB.",
    label_visibility="visible",
)

# ───────────────────────────────────────────────
#                MAIN LOGIC
# ───────────────────────────────────────────────
if uploaded_file is not None:

    image    = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ── preview ───────────────────────────────
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon">🖼️</div>
        <div>
          <div class="card-title">Image Preview</div>
          <div class="card-desc">Uploaded file ready for analysis</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.image(image_np, channels="RGB", use_container_width=True)

    # ── analyze button ────────────────────────
    st.markdown('<div style="margin-top:0.6rem"></div>', unsafe_allow_html=True)
    analyze_btn = st.button("  Analyze Image", type="primary", use_container_width=True)

    # ── run analysis ──────────────────────────
    if analyze_btn:

        with st.spinner("Running forensic analysis… please wait"):
            heatmap_raw                          = generate_heatmap(image_np, model, device)
            label, score, area_ratio, heatmap    = classify_image(heatmap_raw)

        # ── RESULT CARD ─────────────────────────
        st.markdown("""
        <div class="card">
          <div class="card-header">
            <div class="card-icon">📊</div>
            <div>
              <div class="card-title">Analysis Result</div>
              <div class="card-desc">Forensic verdict & confidence metrics</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── badge ─────────────────────────────
        if label == "Tampered":
            st.markdown(
                '<div class="result-badge badge-tampered">'
                '<span class="dot"></span> Tampered Detected'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

            # metrics row
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Region Confidence", f"{score:.3f}", delta=None)
            with col_m2:
                st.metric("Affected Area", f"{area_ratio * 100:.2f}%", delta=None)
            with col_m3:
                st.metric("Regions Found", "1+", delta=None)

            # ── visualisations ──────────────────
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
              <div class="card-header">
                <div class="card-icon">🔬</div>
                <div>
                  <div class="card-title">Visualisations</div>
                  <div class="card-desc">Heatmap overlay & bounding-box localisation</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2, gap="medium")

            with col1:
                st.markdown('<div class="img-label">Tampering Heatmap</div>', unsafe_allow_html=True)
                st.image(overlay_heatmap(image_np, heatmap), use_container_width=True)

            with col2:
                st.markdown('<div class="img-label">Detected Regions</div>', unsafe_allow_html=True)
                st.image(draw_bounding_boxes(image_np, heatmap), use_container_width=True)

        else:   # ── ORIGINAL ─────────────────────
            st.markdown(
                '<div class="result-badge badge-original">'
                '<span class="dot"></span> Original'
                '</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
            st.success("No significant tampering detected in this image.")
            st.caption("Heatmap localisation is not rendered for images classified as original.")

# ───────────────────────────────────────────────
#                   FOOTER
# ───────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
  <div class="footer-text">IMAGE TAMPERING DETECTION & LOCALISATION  ·  POWERED BY DEEP LEARNING  ·  © 2025 </div>
</div>
""", unsafe_allow_html=True)