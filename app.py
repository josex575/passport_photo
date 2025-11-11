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

# Download PDF with multiple copies vertically on 4x6 inch paper
pdf_buf = io.BytesIO()
c = canvas.Canvas(pdf_buf, pagesize=(4*inch, 6*inch))  # 4x6 inch paper, portrait

# Coordinates for photos vertically
y_positions = [4.0*inch, 2.0*inch, 0.0*inch]  # top to bottom with spacing
x_position = 1.0*inch  # center horizontally

# Save temp photo
bordered.save("temp.jpg")

for y in y_positions:
    c.drawImage("temp.jpg", x_position, y, width=2*inch, height=2*inch)

c.showPage()
c.save()
pdf_buf.seek(0)

st.download_button(
    "📥 Download PDF (4x6 paper, vertical photos for reuse)",
    data=pdf_buf,
    file_name="passport_layout.pdf",
    mime="application/pdf"
)
