import streamlit as st
from PIL import Image, ImageDraw
import io
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, mm
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

# Resize to passport size (51 mm x 51 mm)
dpi = 300
size_px = int(51 / 25.4 * dpi)  # 51mm at 300 DPI
passport_photo = cropped_img.resize((size_px, size_px), Image.LANCZOS)

# Draw narrow border line
draw = ImageDraw.Draw(passport_photo)
border_width = 2  # 2 px border
draw.rectangle([0, 0, size_px-1, size_px-1], outline="black", width=border_width)

st.image(passport_photo, caption="Cropped Passport Photo with Border", width=150)

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
c = canvas.Canvas(pdf_buf, pagesize=(6*inch, 4*inch))  # 6x4 inch paper, landscape

# Convert mm to points for PDF (1 mm = 2.83465 pt)
photo_size_pt = 51 * 2.83465

# Coordinates: bottom-left and top-left
x_position = 0  # left edge
y_positions = [0, 4*inch - photo_size_pt]  # bottom and top

# Save temp photo with border
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
