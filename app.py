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
st.title("📸 Passport Studio — Auto Face Crop, Background Whitening, Print PDF")
st.markdown("Auto-detect face (MediaPipe), auto-crop, fine-tune, whiten background, and export JPG/PDF.")

# -------------------------
# Passport presets
# -------------------------
PASSPORT_PRESETS = {
    "India (51×51 mm)": {"mm_w": 51, "mm_h": 51},
    "USA (2×2 inches = 51×51 mm)": {"mm_w": 51, "mm_h": 51},
    "UK (35×45 mm)": {"mm_w": 35, "mm_h": 45},
    "EU (35×45 mm)": {"mm_w": 35, "mm_h": 45},
    "Custom (px)": {"mm_w": None, "mm_h": None},
}

PRINT_DPI = 300  # For print

# -------------------------
# Caching
# -------------------------
@st.cache_data
def load_image_bytes(uploaded_file):
    return uploaded_file.read()

@st.cache_data
def load_image_pil(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_dim = 2500
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return img

@lru_cache(maxsize=1)
def get_mediapipe_models():
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    return mp_face, mp_selfie

# -------------------------
# Helpers
# -------------------------
def pil_to_cv2(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

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

def compress_image_bytes(pil_img, min_kb=20, max_kb=100):
    buf = io.BytesIO()
    quality = 90
    for _ in range(12):
        buf.seek(0)
        pil_img.save(buf, format="JPEG", quality=quality, dpi=(PRINT_DPI, PRINT_DPI))
        size_kb = buf.tell() / 1024
        if min_kb <= size_kb <= max_kb:
            break
        quality = max(10, min(95, quality - 7 if size_kb>max_kb else quality+5))
    buf.seek(0)
    return buf

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Studio Controls")
preset = st.sidebar.selectbox("Passport Size Preset", list(PASSPORT_PRESETS.keys()))
custom_px = None
if preset == "Custom (px)":
    custom_px = st.sidebar.number_input("Custom size (px)", min_value=200, max_value=2000, value=600, step=10)
copies_option = st.sidebar.selectbox("Copies per sheet", ["4 (6x4)", "6 (A4)", "8 (A4)"])
whiten_bg = st.sidebar.checkbox("Auto whiten background", value=True)
allow_manual_tune = st.sidebar.checkbox("Allow fine-tune after auto-crop", value=True)
auto_margin_ratio = st.sidebar.slider("Auto-crop margin ratio", 0.2, 1.5, 0.6, 0.05)
border_px = st.sidebar.number_input("Cut border (px)", min_value=0, max_value=80, value=10)
pdf_page_option = st.sidebar.selectbox("PDF Page Format", ["6x4 (landscape)", "A4 (landscape)"])

# Compute target pixels
if preset != "Custom (px)":
    mm_w = PASSPORT_PRESETS[preset]["mm_w"]
    mm_h = PASSPORT_PRESETS[preset]["mm_h"]
    inches_w = mm_w / 25.4
    inches_h = mm_h / 25.4
    target_px = int(round(max(inches_w, inches_h)*PRINT_DPI))
else:
    target_px = int(custom_px or 600)
target_px = max(350, min(1000, target_px))

# -------------------------
# Main UI
# -------------------------
col1, col2 = st.columns([1,1])
with col1:
    uploaded_file = st.file_uploader("Upload photo (front-facing)", type=["jpg","jpeg","png"])
with col2:
    st.markdown("### Steps:\n1. Upload photo\n2. Auto face crop\n3. Fine-tune if needed\n4. Download JPG/PDF")
if not uploaded_file:
    st.info("Upload a photo to start.")
    st.stop()

image_bytes = load_image_bytes(uploaded_file)
orig_pil = load_image_pil(image_bytes)
st.markdown("### Original Image")
st.image(orig_pil, use_column_width=True)

# Face detection
with st.spinner("Detecting face..."):
    face_box = detect_face_bbox(orig_pil)

if face_box is None:
    st.warning("No face detected. Use manual crop or retry with clearer photo.")
    st.stop()

x, y, w, h = face_box
img_w, img_h = orig_pil.size
auto_box = add_margin_square((x,y,w,h), img_w, img_h, margin_ratio=auto_margin_ratio)

# Fine-tune
cropped = crop_to_box(orig_pil, auto_box)
if allow_manual_tune:
    st.subheader("Fine-tune crop (optional)")
    dx = st.slider("Shift X", -200, 200, 0)
    dy = st.slider("Shift Y", -200, 200, 0)
    scale = st.slider("Scale (%)", 70, 140, 100)
    # apply adjustments
    cx1, cy1, cx2, cy2 = auto_box
    c_w = cx2 - cx1
    c_h = cy2 - cy1
    new_w = int(c_w * scale/100)
    new_h = int(c_h * scale/100)
    center_x = cx1 + c_w//2 + dx
    center_y = cy1 + c_h//2 + dy
    new_x1 = max(0, center_x - new_w//2)
    new_y1 = max(0, center_y - new_h//2)
    new_x2 = min(img_w, new_x1 + new_w)
    new_y2 = min(img_h, new_y1 + new_h)
    crop_box = (new_x1, new_y1, new_x2, new_y2)
    cropped = crop_to_box(orig_pil, crop_box)

# Background whitening
if whiten_bg:
    mask = segment_background_mask(cropped)
    cropped = apply_background_whitening(cropped, mask)

# Resize & compress
cropped = resize_for_passport(cropped, target_px)
cropped.info["dpi"]=(PRINT_DPI,PRINT_DPI)
compressed_buf = compress_image_bytes(cropped, 20,100)

st.markdown("### Final Cropped Passport Photo")
st.image(cropped, width=300)
st.download_button("📥 Download Cropped JPG", data=compressed_buf, file_name="passport_photo.jpg", mime="image/jpeg")

# Layout image
def make_layout_image(pil_photo, copies=4, page="6x4"):
    if page=="6x4":
        page_w_px = int(6*PRINT_DPI)
        page_h_px = int(4*PRINT_DPI)
    else:
        page_w_px = int(11.69*PRINT_DPI)
        page_h_px = int(8.27*PRINT_DPI)
    page_img = Image.new("RGB",(page_w_px,page_h_px),"white")
    bordered = ImageOps.expand(pil_photo, border=border_px, fill="lightgray")
    if copies==4:
        cols=2; rows=2
    elif copies==6:
        cols=3; rows=2
    elif copies==8:
        cols=4; rows=2
    else:
        cols=2; rows=2
    bw, bh = bordered.size
    gap_x = max(20,(page_w_px - cols*bw)//(cols+1))
    gap_y = max(20,(page_h_px - rows*bh)//(rows+1))
    start_x=start_y=0
    for r in range(rows):
        for c in range(cols):
            x=start_x+gap_x + c*(bw+gap_x)
            y=start_y+gap_y + r*(bh+gap_y)
            if x+bw<=page_w_px and y+bh<=page_h_px:
                page_img.paste(bordered,(x,y))
    return page_img

copies = 4 if copies_option.startswith("4") else 6 if copies_option.startswith("6") else 8
page_type = "6x4" if pdf_page_option.startswith("6") else "A4"
layout_img = make_layout_image(cropped, copies=copies, page=page_type)

st.subheader("Printable Layout Preview")
st.image(layout_img, width=600)
layout_buf = io.BytesIO()
layout_img.save(layout_buf, format="JPEG", quality=95, dpi=(PRINT_DPI,PRINT_DPI))
layout_buf.seek(0)
st.download_button("📥 Download Layout JPG", data=layout_buf, file_name="passport_layout.jpg", mime="image/jpeg")

# PDF export
try:
    from reportlab.lib.utils import ImageReader
    pdf_buf = io.BytesIO()
    pdf_w = 6*inch if page_type=="6x4" else 11.69*inch
    pdf_h = 4*inch if page_type=="6x4" else 8.27*inch
    c = canvas.Canvas(pdf_buf, pagesize=(pdf_w,pdf_h))
    img_buf = io.BytesIO()
    layout_img.save(img_buf, format="JPEG", quality=95)
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf),0,0,width=pdf_w,height=pdf_h)
    c.showPage()
    c.save()
    pdf_buf.seek(0)
    st.download_button("📥 Download Print-ready PDF", data=pdf_buf, file_name="passport_layout.pdf", mime="application/pdf")
except Exception as e:
    st.warning(f"PDF generation failed: {e}")

st.success("Ready — download your passport photo and printable layout!")
