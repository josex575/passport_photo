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
    y1 = max(0, y - margin)
    x2 = min(cv_img.shape[1], x + w + margin)
    y2 = min(cv_img.shape[0], y + h + margin)

    face_region = cv_img[y1:y2, x1:x2]
    cropped_pil = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
    return cropped_pil

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = load_image(uploaded_file)
    st.image(img, caption=f"Uploaded image ({img.width}×{img.height}px)", use_container_width=True)

    st.subheader("Step 1: Auto-Detect & Crop Face")

    with st.spinner("Detecting face..."):
        cropped = detect_and_crop_face(img)

    if cropped is None:
        st.error("😕 No face detected. Please upload a clearer photo (front-facing, good lighting).")
    else:
        st.success("✅ Face detected and cropped successfully!")
        cropped = cropped.resize((600, 600), Image.LANCZOS)
        cropped.info["dpi"] = (300, 300)
        cropped_buf = compress_image(cropped)

        # --- Create printable 6×4 layout (4 copies) ---
        layout = Image.new("RGB", (1800, 1200), "white")  # 6×4" @ 300 DPI
        bordered = ImageOps.expand(cropped, border=10, fill="lightgray")
        margin_x, margin_y = 150, 100
        for row in range(2):
            for col in range(2):
                x = margin_x + col * (bordered.width + margin_x)
                y = margin_y + row * (bordered.height + margin_y)
                layout.paste(bordered, (x, y))

        layout_buf = io.BytesIO()
        layout.save(layout_buf, format="JPEG", quality=95, dpi=(300, 300))
        layout_buf.seek(0)

        # --- Display results ---
        st.subheader("Cropped Passport Photo (600×600)")
        st.image(cropped, use_container_width=False)

        st.download_button("📥 Download Cropped Photo", cropped_buf, "passport_photo.jpg", "image/jpeg")

        st.subheader("Printable 6×4 Layout (4 Photos)")
        st.image(layout, use_container_width=False)
        st.download_button("📥 Download 6×4 Layout", layout_buf, "passport_layout.jpg", "image/jpeg")
else:
    st.info("⬆️ Upload a photo above to start.")
