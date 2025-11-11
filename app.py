import streamlit as st
from PIL import Image, ImageOps
import io
from reportlab.pdfgen import canvas
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

# Interactive cropper
st.markdown("### Drag and resize the rectangle to crop the passport photo")
cropped_img = st_cropper(
    image,
    realtime_update=True,
    box_color="#FF0000",
    aspect_ratio=(1,1),  # square aspect ratio
    return_type="image"  # PIL.Image
)

# Resize to passport size (600x600 px)
cropped_resized = cropped_img.resize((600,600), Image.LANCZOS)

# Add light border
bordered = ImageOps.expand(cropped_resized, border=20, fill="white")
st.image(bordered, caption="Cropped Passport Photo Preview", width=300)

# Download JPG
buf_jpg = io.BytesIO()
bordered.save(buf_jpg, format="JPEG", quality=90)
buf_jpg.seek(0)
st.download_button("📥 Download JPG", data=buf_jpg, file_name="passport_photo.jpg", mime="image/jpeg")

# Download PDF with multiple copies on top of 6x4 paper for reuse after cutting
pdf_buf = io.BytesIO()
c = canvas.Canvas(pdf_buf, pagesize=(6*inch, 4*inch))  # 6x4 inch paper

# Coordinates for three photos on top (can reuse paper)
x_positions = [0.25*inch, 2.25*inch, 4.25*inch]  # left positions with margin
# Top margin: 0.5 inch, each photo height 2 inch
y_position = 4*inch - 2.5*inch

# Save temp photo
bordered.save("temp.jpg")

for x in x_positions:
    c.drawImage("temp.jpg", x, y_position, width=2*inch, height=2*inch)

c.showPage()
c.save()
pdf_buf.seek(0)

st.download_button(
    "📥 Download PDF (6x4 paper, multiple photos for reuse)",
    data=pdf_buf,
    file_name="passport_layout.pdf",
    mime="application/pdf"
)
