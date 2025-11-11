import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import io
import numpy as np
import cv2
import mediapipe as mp
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import landscape, A4
from functools import lru_cache

st.set_page_config(page_title="Passport Studio", page_icon="📸", layout="wide")
st.title("📸 Passport Studio — Auto Face Crop & Print PDF")
st.markdown("Upload a photo, auto-detect face, crop, whiten background, and generate passport-ready JPG/PDF.")

PASSPORT_PRESETS = {
    "India (51×51 mm)": {"mm_w": 51, "mm_h": 51},
    "USA (2×2 inches = 51×51 mm)": {"mm_w": 51, "mm_h": 51},
    "UK (35×45 mm)": {"mm_w": 35, "mm_h": 45},
    "EU (35×45 mm)": {"mm_w": 35, "mm_h": 45},
    "Custom (px)": {"mm_w": None, "mm_h": None},
}

PRINT_DPI = 300

@st.cache_data
def load_image_bytes(uploaded_file):
    return uploaded_file.read()

@st.cache_data
def load_image_pil(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_dim = 2500
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width*ratio), int(img.height*ratio)), Image.LANCZOS)
    return img

@lru_cache(maxsize=1)
def get_mediapipe_models():
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    return mp_face, mp_selfie

def pil_to_cv2(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def detect_face_bbox(pil_img):
    mp_face, _ = get_mediapipe_models()
    img_cv = pil_to_cv2(pil_img)
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)
    if not results.detections:
        return None
    h, w = img_cv.shape[:2]
    best = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width *
               d.location_data.relative_bounding_box.height)
    bboxC = best.location_data.relative_bounding_box
    x = int(bboxC.xmin * w)
    y = int(bboxC.ymin * h)
    bw = int(bboxC.width * w)
    bh = int(bboxC.height * h)
    return x, y, bw, bh

def segment_background_mask(pil_img):
    _, mp_selfie = get_mediapipe_models()
    img_cv = pil_to_cv2(pil_img)
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    results = mp_selfie.process(rgb)
    if results.segmentation_mask is None:
        return Image.new("L", pil_img.size, 255)
    seg_resized = cv2.resize(results.segmentation_mask, pil_img.size)
    mask = (seg_resized > 0.5).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask).convert("L")
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=3))
    return mask_pil

def apply_background_whitening(pil_img, mask_pil):
    white_bg = Image.new("RGB", pil_img.size, (255,255,255))
    return Image.composite(pil_img, white_bg, mask_pil)

def add_margin_square(box, img_w, img_h, margin_ratio=0.6):
    x, y, w, h = box
    margin = int(h * margin_ratio)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_w, x + w + margin)
    y2 = min(img_h, y + h + margin)
    bw = x2 - x1
    bh = y2 - y1
    side = max(bw, bh)
    cx = x1 + bw // 2
    cy = y1 + bh // 2
    sx1 = max(0, cx - side // 2)
    sy1 = max(0, cy - side // 2)
    sx2 = min(img_w, sx1 + side)
    sy2 = min(img_h, sy1 + side)
    sx1 = max(0, sx2 - side)
    sy1 = max(0, sy2 - side)
    return (sx1, sy1, sx2, sy2)

def crop_to_box(pil_img, box):
    return pil_img.crop(box)

def resize_for_passport(pil_img, target_px):
    return pil_img.resize((target_px, target_px), Image.LANCZOS)

uploaded_file = st.file_uploader("Upload front-facing photo", type=["jpg","jpeg","png"])
if not uploaded_file:
    st.info("Upload a photo to start.")
    st.stop()

image_bytes = load_image_bytes(uploaded_file)
orig_pil = load_image_pil(image_bytes)
st.image(orig_pil, caption="Original", use_column_width=True)

face_box = detect_face_bbox(orig_pil)
if face_box is None:
    st.warning("No face detected. Please try another photo.")
    st.stop()

x, y, w, h = face_box
img_w, img_h = orig_pil.size
auto_box = add_margin_square((x,y,w,h), img_w, img_h)

cropped = crop_to_box(orig_pil, auto_box)
mask = segment_background_mask(cropped)
cropped = apply_background_whitening(cropped, mask)
cropped = resize_for_passport(cropped, 600)

buf = io.BytesIO()
cropped.save(buf, format="JPEG", quality=90, dpi=(PRINT_DPI, PRINT_DPI))
buf.seek(0)

st.image(cropped, caption="Cropped Passport Photo", width=300)
st.download_button("📥 Download JPG", data=buf, file_name="passport_photo.jpg", mime="image/jpeg")
