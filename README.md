# Image Tampering Detection and Localization

A deep learning–based system to **detect image tampering** and **localize manipulated regions** using patch-level analysis and visual heatmaps.  
The project is designed as an **end-to-end ML application** with model training, inference, and an interactive web interface.

---

## 🔍 Problem Statement

Digital images can be easily manipulated using modern editing tools, making it difficult to verify their authenticity.  
This project aims to:

- Detect whether an image is **tampered or original**
- Localize suspicious regions if tampering is detected
- Provide visual explanations using **heatmaps and bounding boxes**

---

## 🎯 Key Features

- Patch-based image analysis using deep learning
- Pretrained CNN backbone for robust feature extraction
- Heatmap-based localization of manipulated regions
- Automatic classification: **Original / Tampered**
- Interactive web interface using Streamlit
- GPU-accelerated inference (if available)

---

## 🧠 Methodology Overview

1. **Patch Extraction**
   - The input image is divided into overlapping patches
   - Each patch is independently analyzed

2. **Feature Extraction**
   - A pretrained CNN (ResNet-based) extracts deep features
   - Transfer learning improves performance on small datasets

3. **Patch Classification**
   - Each patch is classified as normal or suspicious

4. **Heatmap Generation**
   - Patch predictions are aggregated into a spatial heatmap

5. **Decision Logic**
   - Statistical analysis of the heatmap determines:
     - Global image label (Original / Tampered)
     - Localized suspicious regions

6. **Visualization**
   - Heatmap overlay on original image
   - Bounding boxes around detected regions

---
## 🖼 How the Application Works
1. User uploads an image (JPG / JPEG / PNG)
2. Image is divided into overlapping patches
3. Each patch is analyzed using a deep learning model
4. Patch-level predictions are aggregated into a heatmap
5. Statistical analysis determines:
   - Whether the image is tampered
   - The location of suspicious regions
6. Results are visualized using:
   - Heatmap overlay
   - Bounding boxes (only for tampered images)
---
## 🧩 Technologies Used

Programming Language:
- Python 3.10.11

Deep Learning & ML:
- PyTorch
- Torchvision 

Image Processing:
- OpenCV
- NumPy
- Pillow (PIL)

Web Application:
- Streamlit
---
## ⚙️ Installation

Repository Setup:
- Clone the repository from GitHub
- Navigate into the project directory

Virtual Environment:
- Create a Python virtual environment
- Activate the virtual environment

Dependencies:
- Install required Python packages using requirements.txt

Application Run:
- Launch the Streamlit application
- Access the app via the local browser URL
``` bash 
# Clone repository 
git clone https://github.com/<your-username>/image-tampering-detection.git
cd image-tampering-detection
```
``` bash 
# Create virtual environment
python -m venv venv
```
```bash
# Activate virtual environment

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```
```bash
# Install dependencies
pip install -r requirements.txt
```
```bash
# Run the application
streamlit run app.py
```
---
## Output Interpretation 

**ORIGINAL**
- No dominant anomalous regions detected
- Heatmap and bounding boxes are hidden
- Image is likely authentic
  
**TAMPERED**
- Suspicious regions detected
- Heatmap highlights manipulated areas
- Bounding boxes indicate localized tampering
- Confidence and affected area are displayed

--- 
The results are probabilistic and depend on image quality,
dataset characteristics, and tampering techniques.

