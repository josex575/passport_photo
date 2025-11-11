import streamlit as st
from PIL import Image, ImageOps
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch
from streamlit_cropper import st_cropper

st.set_page_config(page_title="Passport Studio", page_icon="📸", layout="wide")
st.title("📸 Passport Studio (Interactive Crop, Print Ready)")

# Upload photo
uploaded_file = st.file_uploader("Upload your photo", type=["jpg","jpeg","png"])
if not uploaded_file:
    st.info("Please upload a photo to start.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Original Photo", use_column_width=True)

# ---------------------------
# Interactive cropper
# ---------------------------
st.markdown("### Drag and resize the rectangle to crop the passport photo")
cropped_img = st_cropper(
    image,
    realtime_update=True,
    box_color="#FF0000",
    aspect_ratio=(1,1),  # fixed 1:1 square
    return_type="image"
)

# Resize to passport size (51x51 mm ~ 600x600 px)
cropped_resized = cropped_img.resize((600,600), Image.LANCZOS)

# Add light border
bordered = ImageOps.expand(cropped_resized, border=20, fill="white")

st.image(bordered, caption="Cropped Passport Photo Preview", width=300)

# ---------------------------
# Download JPG
# ---------------------------
buf_jpg = io.BytesIO()
bordered.save(buf_jpg, format="JPEG", quality=90)
buf_jpg.seek(0)
st.download_button("📥 Download JPG", data=buf_jpg, file_name="passport_photo.jpg", mime="image/jpeg")

# ---------------------------
# Download PDF for 6x4 layout
# ---------------------------
pdf_buf = io.BytesIO()
c = canvas.Canvas(pdf_buf, pagesize=landscape(A4))
page_w, page_h = landscape(A4)
img_w, img_h = bordered.size
# Place image top-left, leave rest blank
x_pos = inch
y_pos = page_h - img_h - inch
bordered.save("temp.jpg")  # reportlab needs file path
c.drawImage("temp.jpg", x_pos, y_pos, width=img_w, height=img_h)
c.showPage()
c.save()
pdf_buf.seek(0)
st.download_button("📥 Download PDF (6x4 paper)", data=pdf_buf, file_name="passport_layout.pdf", mime="application/pdf")
