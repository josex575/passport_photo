import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
import io
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, mm
from streamlit_cropper import st_cropper

# Custom CSS for professional app background (light gray)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Passport Studio", page_icon="📸", layout="wide")
st.title("📸 Passport Studio (Interactive Crop, Print Ready)")

# Upload photo
uploaded_file = st.file_uploader("Upload your photo", type=["jpg","jpeg","png"])
if not uploaded_file:
    st.info("Please upload a photo to start.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Original Photo", use_column_width=True)

# Option to adjust color tone
adjust_tone = st.slider("Adjust Color Tone", 0.5, 2.0, 1.0, 0.1)

# Interactive cropper
st.markdown("### Drag and resize the rectangle to crop the passport photo")
cropped_img = st_cropper(
    image,
    realtime_update=True,
    box_color="#FF0000",
    aspect_ratio=(1,1),
    return_type="image"
)

# Resize to passport size (51 mm x 51 mm)
dpi = 300
size_px = int(51 / 25.4 * dpi)
passport_photo = cropped_img.resize((size_px, size_px), Image.LANCZOS)

# Adjust color tone
enhancer = ImageEnhance.Color(passport_photo)
passport_photo = enhancer.enhance(adjust_tone)

# Draw narrow border line
draw = ImageDraw.Draw(passport_photo)
border_width = 2
draw.rectangle([0, 0, size_px-1, size_px-1], outline="black", width=border_width)

st.image(passport_photo, caption="Processed Passport Photo", width=150)

# Download individual JPG photos (two copies)
buf_jpg1 = io.BytesIO()
buf_jpg2 = io.BytesIO()
passport_photo.save(buf_jpg1, format="JPEG", quality=95, dpi=(dpi,dpi))
passport_photo.save(buf_jpg2, format="JPEG", quality=95, dpi=(dpi,dpi))
buf_jpg1.seek(0)
buf_jpg2.seek(0)

st.download_button("📥 Download Photo 1 (JPG)", data=buf_jpg1, file_name="passport_photo_1.jpg", mime="image/jpeg")
st.download_button("📥 Download Photo 2 (JPG)", data=buf_jpg2, file_name="passport_photo_2.jpg", mime="image/jpeg")

# Download PDF with 2 photos side by side at left edge
pdf_buf = io.BytesIO()
c = canvas.Canvas(pdf_buf, pagesize=(6*inch, 4*inch))
photo_size_pt = 51 * 2.83465
x_position = 0
y_positions = [0, 4*inch - photo_size_pt]
passport_photo.save("temp.jpg")
for y in y_positions:
    c.drawImage("temp.jpg", x_position, y, width=photo_size_pt, height=photo_size_pt)
c.showPage()
c.save()
pdf_buf.seek(0)

st.download_button(
    "📥 Download PDF (4x6 paper, 2 photos at left edge with border)",
    data=pdf_buf,
    file_name="passport_layout.pdf",
    mime="application/pdf"
)
