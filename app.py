import streamlit as st
from PIL import Image, ImageOps
import cv2
import numpy as np
import io

st.set_page_config(page_title="Passport Photo Cropper", page_icon="📸", layout="centered")

st.title("📸 Passport Photo Auto Cropper & Layout Generator")
st.markdown("""
Upload a photo, and this app will **automatically detect your face**, crop it to passport standards,
and create a printable 6×4 sheet (4 copies).

**Passport Photo Specs:**
- Size: 51 × 51 mm (2 × 2 inches)
- Digital size: 20–100 KB
- Resolution: 600×600 px (24-bit color)
- DPI: 300
""")

# --- Cache image loading and face detection ---
@st.cache_data
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    if max(img.size) > 2000:
        ratio = 2000 / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return img

def compress_image(img, min_kb=20, max_kb=100):
    quality = 90
    buf = io.BytesIO()
    for _ in range(10):
        buf.seek(0)
        img.save(buf, format="JPEG", quality=quality)
        size_kb = buf.tell() / 1024
        if min_kb <= size_kb <= max_kb:
            break
        quality -= 5 if size_kb > max_kb else -5
    buf.seek(0)
    return buf

def detect_and_crop_face(pil_img):
    """Detect face using OpenCV and crop region."""
    # Convert to OpenCV format
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None  # No face found

    # Choose the largest detected face (most likely the main subject)
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    # Add margin for shoulders/headroom
    margin = int(0.6 * h)
    x1 = max(0, x - margin)
