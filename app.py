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

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="Passport Studio", page_icon="📸", layout="wide")
st.title("📸 Passport Studio — Auto Face Crop, Background Whitening, Print PDF")
st.markdown("Auto-detect face (MediaPipe), auto-crop, fine-tune, whiten background, and export JPG/PDF ready for printing.")

# -------------------------
# Helper: passport presets
# -------------------------
PASSPORT_PRESETS = {
    "India (51×51 mm)": {"mm_w": 51, "mm_h": 51},
    "USA (2×2 inches = 51×51 mm)": {"mm_w": 51, "mm_h": 51},
    "UK (35×45 mm)": {"mm_w": 35, "mm_h": 45},
    "EU (35×45 mm)": {"mm_w": 35, "mm_h": 45},
    "Custom (px)": {"mm_w": None, "mm_h": None},
}

# dpi to use for print-ready images
PRINT_DPI = 300

# -------------------------
# Caching utilities
# -------------------------
@st.cache_data
def load_image_bytes(uploaded_file):
    return uploaded_file.read()

@st.cache_data
def load_image_pil(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize very large images to a manageable max dimension (to avoid memory issues)
    max_dim = 2500
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return img

# Cache models (MediaPipe)
@lru_cache(maxsize=1)
def get_mediapipe_models():
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    return mp_face, mp_selfie

# -------------------------
# Image processing helpers
# -------------------------
def pil_to_cv2(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def detect_face_bbox(pil_img):
    """Return bounding box (x,y,w,h) for largest detected face in image or None."""
    mp_face, _ = get_mediapipe_models()
    img_cv = pil_to_cv2(pil_img)
    # Convert to RGB for mediapipe
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)
    if not results.detections:
        return None
    # Find largest detection
    h, w = img_cv.shape[:2]
    best = None
    best_area = 0
    for det in results.detections:
        bboxC = det.location_data.relative_bounding_box
        x = int(bboxC.xmin * w)
        y = int(bboxC.ymin * h)
        bw = int(bboxC.width * w)
        bh = int(bboxC.height * h)
        area = bw * bh
        if area > best_area:
            best_area = area
            best = (x, y, bw, bh)
    return best

def segment_background_mask(pil_img):
    """Return mask (PIL L mode) where background=0, foreground=255 using MediaPipe SelfieSegmentation."""
    _, mp_selfie = get_mediapipe_models()
    img_cv = pil_to_cv2(pil_img)
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    results = mp_selfie.process(rgb)
    if results.segmentation_mask is None:
        # fallback: full foreground
        mask = Image.new("L", pil_img.size, 255)
        return mask
    seg = results.segmentation_mask
    # seg is float32 HxW with values 0..1. Resize to original image if sizes differ.
    seg_resized = cv2.resize(seg, pil_img.size)
    # Create mask thresholded
    mask = (seg_resized > 0.5).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask).convert("L")
    # Optionally blur edges for softer cutouts
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=3))
    return mask_pil

def apply_background_whitening(pil_img, mask_pil):
    """Composite image over white using the mask. mask==255 keeps foreground."""
    white_bg = Image.new("RGB", pil_img.size, (255,255,255))
    result = Image.composite(pil_img, white_bg, mask_pil)
    return result

def add_margin_square(box, img_w, img_h, margin_ratio=0.6):
    """Given box (x,y,w,h), produce expanded square box with margin ratio relative to face height."""
    x, y, w, h = box
    margin = int(h * margin_ratio)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_w, x + w + margin)
    y2 = min(img_h, y + h + margin)
    # convert to square by expanding shorter side
    bw = x2 - x1
    bh = y2 - y1
    side = max(bw, bh)
    # center current box in new square
    cx = x1 + bw // 2
    cy = y1 + bh // 2
    sx1 = max(0, cx - side // 2)
    sy1 = max(0, cy - side // 2)
    sx2 = min(img_w, sx1 + side)
    sy2 = min(img_h, sy1 + side)
    # if clamp causes size < side (edge), re-adjust
    sx1 = max(0, sx2 - side)
    sy1 = max(0, sy2 - side)
    return (sx1, sy1, sx2, sy2)

def crop_to_box(pil_img, box):
    """Crop PIL image to box (x1,y1,x2,y2)."""
    return pil_img.crop(box)

def resize_for_passport(pil_img, target_px):
    return pil_img.resize((target_px, target_px), Image.LANCZOS)

def compress_image_bytes(pil_img, min_kb=20, max_kb=100):
    """Return BytesIO JPEG compressed between size limits if possible."""
    buf = io.BytesIO()
    quality = 90
    for _ in range(12):
        buf.seek(0)
        pil_img.save(buf, format="JPEG", quality=quality, dpi=(PRINT_DPI, PRINT_DPI))
        size_kb = buf.tell() / 1024
        if min_kb <= size_kb <= max_kb:
            break
        if size_kb > max_kb:
            quality = max(10, quality - 7)
        else:
            quality = min(95, quality + 5)
    buf.seek(0)
    return buf

# -------------------------
# UI: Sidebar controls
# -------------------------
st.sidebar.header("Studio Controls")
preset = st.sidebar.selectbox("Passport Size Preset", list(PASSPORT_PRESETS.keys()))
custom_px = None
if preset == "Custom (px)":
    custom_px = st.sidebar.number_input("Custom size (px)", min_value=200, max_value=2000, value=600, step=10)

copies_option = st.sidebar.selectbox("Copies per sheet (layout)", ["4 (6×4)", "6 (A4)", "8 (A4)"])
whiten_bg = st.sidebar.checkbox("Auto whiten background", value=True)
allow_manual_tune = st.sidebar.checkbox("Allow manual fine-tune after auto-crop", value=True)
auto_margin_ratio = st.sidebar.slider("Auto-crop margin (ratio of face height)", 0.2, 1.5, 0.6, step=0.05)
pdf_page_option = st.sidebar.selectbox("Export format for PDF", ["6x4 (landscape)", "A4 (landscape)"])
download_jpeg_btn_text = st.sidebar.text_input("Download JPEG button text", "Download Cropped JPG", max_chars=40)

# Target px based on preset
if preset != "Custom (px)":
    # default target px: 300 dpi * inches(mm)/25.4
    mm_w = PASSPORT_PRESETS[preset]["mm_w"]
    mm_h = PASSPORT_PRESETS[preset]["mm_h"]
    # compute px using PRINT_DPI
    # Prefer square: choose width in mm (we treat target_px as width in px for square)
    # Convert mm to inches: /25.4
    if mm_w and mm_h:
        inches_w = mm_w / 25.4
        inches_h = mm_h / 25.4
        # choose px by average inches scaled to DPI (we will output square - use max of w,h)
        target_inches = max(inches_w, inches_h)
        target_px = int(round(target_inches * PRINT_DPI))
    else:
        target_px = 600
else:
    target_px = int(custom_px or 600)

# Safety minimum
target_px = max(350, min(1000, target_px))

# -------------------------
# Main UI: upload + processing
# -------------------------
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload photo (front-facing, good lighting)", type=["jpg", "jpeg", "png"])
with col2:
    st.markdown("### Steps")
    st.markdown("1. Upload a photo → 2. App auto-detects face & crops → 3. Use fine-tune if needed → 4. Download JPG or PDF.")

if not uploaded_file:
    st.info("Upload a photo to get started.")
    st.stop()

# Load image
image_bytes = load_image_bytes(uploaded_file)
orig_pil = load_image_pil(image_bytes)
st.markdown("### Original Image")
st.image(orig_pil, use_column_width=True)

# Face detection
with st.spinner("Detecting face..."):
    face_box = detect_face_bbox(orig_pil)

if face_box is None:
    st.warning("No face detected automatically. You can use manual crop to select region.")
    # Provide manual crop controls directly (fallback)
    st.subheader("Manual crop (no face detected)")
    left = st.number_input("Left (px)", 0, orig_pil.width-2, 0)
    top = st.number_input("Top (px)", 0, orig_pil.height-2, 0)
    right = st.number_input("Right (px)", 1, orig_pil.width-1, orig_pil.width)
    bottom = st.number_input("Bottom (px)", 1, orig_pil.height-1, orig_pil.height)
    if st.button("Crop & Generate (manual)"):
        crop_box = (left, top, right, bottom)
        cropped = crop_to_box(orig_pil, crop_box)
        # proceed to post-processing below
    else:
        st.stop()
else:
    st.success("Face detected.")
    x, y, w, h = face_box
    # compute expanded square with margin
    img_w, img_h = orig_pil.size
    auto_box = add_margin_square((x,y,w,h), img_w, img_h, margin_ratio=auto_margin_ratio)
    st.markdown("#### Auto-crop preview")
    st.image(orig_pil.crop(auto_box), caption="Auto-cropped region", use_column_width=False)

    # manual fine-tune
    if allow_manual_tune:
        st.subheader("Fine-tune auto-crop")
        st.markdown("Adjust the crop window offsets (pixels) to tweak position or scale.")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            dx = st.slider("Shift X (left/right)", -200, 200, 0)
            scale = st.slider("Scale (percent)", 70, 140, 100)
        with col_b:
            dy = st.slider("Shift Y (up/down)", -200, 200, 0)
            margin_adj = st.slider("Margin ratio", 20, 150, int(auto_margin_ratio*100))
        with col_c:
            border_px = st.number_input("Cut border (px)", min_value=0, max_value=80, value=10)
            preview_size = st.selectbox("Preview size", ["Small", "Medium", "Large"])
        # apply adjustments
        cx1, cy1, cx2, cy2 = auto_box
        c_w = cx2 - cx1
        c_h = cy2 - cy1
        # scale
        s = scale / 100.0
        new_w = int(c_w * s)
        new_h = int(c_h * s)
        center_x = cx1 + c_w//2 + dx
        center_y = cy1 + c_h//2 + dy
        new_x1 = max(0, center_x - new_w//2)
        new_y1 = max(0, center_y - new_h//2)
        new_x2 = min(img_w, new_x1 + new_w)
        new_y2 = min(img_h, new_y1 + new_h)
        crop_box = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
        st.markdown("Preview after fine-tune")
        st.image(orig_pil.crop(crop_box), width=300)
        if st.button("Apply Crop"):
            cropped = crop_to_box(orig_pil, crop_box)
        else:
            st.stop()
    else:
        cropped = crop_to_box(orig_pil, auto_box)

# At this point `cropped` should be set
# Safety check
if 'cropped' not in locals() or cropped is None:
    st.error("Crop failed. Try a different photo or adjust the manual controls.")
    st.stop()

# Optional background whitening
if whiten_bg:
    with st.spinner("Whitening background..."):
        mask = segment_background_mask(cropped)
        cropped = apply_background_whitening(cropped, mask)

# Resize to passport target (square)
cropped = resize_for_passport(cropped, target_px)
st.markdown("### Final Cropped Passport Photo")
st.image(cropped, width=300)
# Ensure 24-bit (RGB) and DPI set (will be applied at save)
cropped.info["dpi"] = (PRINT_DPI, PRINT_DPI)

# Compress and prepare download bytes
compressed_buf = compress_image_bytes(cropped, min_kb=20, max_kb=100)

st.download_button(download_jpeg_btn_text, data=compressed_buf, file_name="passport_photo.jpg", mime="image/jpeg")

# -------------------------
# Generate print layout images & PDF
# -------------------------
def make_layout_image(pil_photo, copies=4, page="6x4"):
    """Return a PIL image with copies placed on a print canvas (6x4 or A4)."""
    if page == "6x4":
        page_w_px = int(6 * PRINT_DPI)
        page_h_px = int(4 * PRINT_DPI)
    else:  # A4 landscape
        page_w_px = int(A4[1])  # A4[1] is width in points; convert points->inches? reportlab handles PDF sizes; here we'll use pixels approximated:
        # approximate: A4 (landscape) in inches = 11.69 x 8.27 ; use 300 dpi
        page_w_px = int(11.69 * PRINT_DPI)
        page_h_px = int(8.27 * PRINT_DPI)
    page_img = Image.new("RGB", (page_w_px, page_h_px), "white")
    bordered = ImageOps.expand(pil_photo, border=border_px if 'border_px' in locals() else 10, fill="lightgray")

    # Layout strategy
    if copies == 4 and page == "6x4":
        # 2x2 grid
        cols = 2; rows = 2
    elif copies == 6 and page == "A4":
        cols = 3; rows = 2
    elif copies == 8 and page == "A4":
        cols = 4; rows = 2
    else:
        # default to 2x2
        cols = 2; rows = 2

    # compute spacing
    bw, bh = bordered.size
    # center grid
    total_w = cols * bw
    total_h = rows * bh
    # spacing between
    if cols > 1:
        gap_x = max(20, (page_w_px - total_w) // (cols + 1))
    else:
        gap_x = (page_w_px - total_w) // 2
    if rows > 1:
        gap_y = max(20, (page_h_px - total_h) // (rows + 1))
    else:
        gap_y = (page_h_px - total_h) // 2

    start_x = gap_x
    start_y = gap_y
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * (bw + gap_x)
            y = start_y + r * (bh + gap_y)
            if x + bw <= page_w_px and y + bh <= page_h_px:
                page_img.paste(bordered, (x, y))
    return page_img

# Choose copies based on selected layout
copies = 4
if copies_option.startswith("6"):
    copies = 6
elif copies_option.startswith("8"):
    copies = 8

page_type = "6x4" if pdf_page_option.startswith("6x4") else "A4"
layout_img = make_layout_image(cropped, copies=copies, page=page_type)

st.subheader("Printable Layout Preview")
st.image(layout_img, use_column_width=False, width=600)

# Prepare image bytes for download
layout_buf = io.BytesIO()
layout_img.save(layout_buf, format="JPEG", quality=95, dpi=(PRINT_DPI, PRINT_DPI))
layout_buf.seek(0)
st.download_button("📥 Download Layout JPG", data=layout_buf, file_name="passport_layout.jpg", mime="image/jpeg")

# Generate PDF
def create_pdf_from_image(pil_page_img, page_choice="6x4"):
    """Return PDF as bytes containing the pil_page_img placed on a page of appropriate size."""
    buf = io.BytesIO()
    if page_choice == "6x4":
        # Create PDF with 6x4 inches page
        pdf_w = 6 * inch
        pdf_h = 4 * inch
        c = canvas.Canvas(buf, pagesize=(pdf_w, pdf_h))
        # Draw the PIL image into PDF. Need to convert to a temporary bytes buffer.
        img_buf = io.BytesIO()
        pil_page_img.save(img_buf, format="JPEG", quality=95)
        img_buf.seek(0)
        # Place full-bleed (keeping aspect)
        c.drawImage(ImageReader(img_buf), 0, 0, width=pdf_w, height=pdf_h)
        c.showPage()
        c.save()
    else:
        # A4 landscape
        pdf_w = 11.69 * inch
        pdf_h = 8.27 * inch
        c = canvas.Canvas(buf, pagesize=(pdf_w, pdf_h))
        img_buf = io.BytesIO()
        pil_page_img.save(img_buf, format="JPEG", quality=95)
        img_buf.seek(0)
        # Fit into A4
        c.drawImage(ImageReader(img_buf), 0, 0, width=pdf_w, height=pdf_h)
        c.showPage()
        c.save()
    buf.seek(0)
    return buf

# ReportLab ImageReader needs import inside because sometimes reportlab.ImageReader isn't available early
try:
    from reportlab.lib.utils import ImageReader
except Exception:
    # fallback: define wrapper using temporary file - but on Streamlit Cloud reportlab ImageReader should be available
    ImageReader = None

# Create PDF bytes
pdf_buf = None
try:
    if ImageReader is not None:
        pdf_buf = create_pdf_from_image(layout_img, page_choice=page_type)
except Exception as e:
    st.warning(f"PDF generation failed: {e}")

if pdf_buf:
    st.download_button("📥 Download Print-ready PDF", data=pdf_buf, file_name="passport_layout.pdf", mime="application/pdf")
else:
    st.info("PDF generation not available in this environment; you can download the JPG and convert to PDF locally.")

st.success("Ready — download your passport photo and layout. Print on 6×4 or A4 paper and cut along the light borders.")
