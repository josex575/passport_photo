import streamlit as st
from PIL import Image, ImageOps
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch

st.set_page_config(page_title="Passport Studio", page_icon="📸", layout="wide")
st.title("📸 Passport Studio (Manual Crop, Print Ready)")

# Upload image
uploaded_file = st.file_uploader("Upload your photo", type=["jpg","jpeg","png"])
if not uploaded_file:
    st.info("Please upload a photo to start.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Original Photo", use_column_width=True)

# Manual crop selection using slider
st.markdown("### Crop the image manually")
col1, col2 = st.columns(2)

with col1:
    left = st.slider("Left", 0, image.width, 0)
    right = st.slider("Right", 0, image.width, image.width)
with col2:
    top = st.slider("Top", 0, image.height, 0)
    bottom = st.slider("Bottom", 0, image.height, image.height)

if st.button("Crop & Resize"):
    # Crop
    cropped = image.crop((left, top, right, bottom))
    # Resize to passport size (51x51 mm approx 600x600 px)
    cropped = cropped.resize((600, 600), Image.LANCZOS)
    # Add light border for print layout
    bordered = ImageOps.expand(cropped, border=20, fill="white")
    
    st.image(bordered, caption="Cropped Passport Photo Preview", width=300)
    
    # Save JPG
    buf_jpg = io.BytesIO()
    bordered.save(buf_jpg, format="JPEG", quality=90)
    buf_jpg.seek(0)
    st.download_button("📥 Download JPG", data=buf_jpg, file_name="passport_photo.jpg", mime="image/jpeg")
    
    # Save PDF with 6x4 paper layout
    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=landscape(A4))
    page_w, page_h = landscape(A4)
    img_w, img_h = bordered.size
    # Place image top-left corner, leaving rest blank
    x_pos = inch
    y_pos = page_h - img_h - inch
    bordered.save("temp.jpg")  # reportlab needs file path
    c.drawImage("temp.jpg", x_pos, y_pos, width=img_w, height=img_h)
    c.showPage()
    c.save()
    pdf_buf.seek(0)
    
    st.download_button("📥 Download PDF (6x4 paper)", data=pdf_buf, file_name="passport_layout.pdf", mime="application/pdf")
