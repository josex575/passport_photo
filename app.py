import streamlit as st
from PIL import Image, ImageOps
import io

st.set_page_config(page_title="Passport Photo Cropper", page_icon="📸", layout="centered")

st.title("📸 Passport Photo Cropper & Layout Generator")
st.markdown("""
Upload a photo, crop it to your face, and generate passport photos that meet official standards.

**Passport Photo Requirements:**
- Size: 51 × 51 mm (2 × 2 inches)
- File size: 20–100 KB
- Resolution: 350–1000 px (we’ll use 600×600)
- Bit Depth: 24-bit color
- DPI: 300
- Output: single cropped photo + printable 6×4 layout (4 photos)
""")

# Image upload
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

def compress_image(img, min_kb=20, max_kb=100):
    """Compress image to meet KB limits."""
    quality = 95
    buf = io.BytesIO()
    while True:
        buf.seek(0)
        img.save(buf, format="JPEG", quality=quality)
        size_kb = buf.tell() / 1024
        if min_kb <= size_kb <= max_kb:
            break
        if size_kb > max_kb:
            quality -= 5
        else:
            quality += 5
        if quality < 10 or quality > 95:
            break
    buf.seek(0)
    return buf

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    # Let user crop using Streamlit's built-in crop box
    st.subheader("Step 1: Crop your photo")
    cropped_img = st.image(img, caption="Uploaded Image", use_container_width=True)

    # Ask for manual crop range
    st.markdown("### Adjust Crop (Optional)")
    left = st.slider("Left", 0, img.width, 0)
    top = st.slider("Top", 0, img.height, 0)
    right = st.slider("Right", 0, img.width, img.width)
    bottom = st.slider("Bottom", 0, img.height, img.height)

    if st.button("Crop & Generate"):
        # Perform crop
        img_cropped = img.crop((left, top, right, bottom))
        img_cropped = img_cropped.resize((600, 600), Image.LANCZOS)
        img_cropped.info["dpi"] = (300, 300)

        # Compress
        cropped_buf = compress_image(img_cropped)

        # --- Create 6x4 layout with 4 copies ---
        layout = Image.new("RGB", (1800, 1200), "white")  # 6x4" @ 300 DPI
        bordered = ImageOps.expand(img_cropped, border=10, fill="lightgray")

        margin_x, margin_y = 150, 100
        for row in range(2):
            for col in range(2):
                x = margin_x + col * (bordered.width + margin_x)
                y = margin_y + row * (bordered.height + margin_y)
                layout.paste(bordered, (x, y))

        # Save images in memory
        layout_buf = io.BytesIO()
        layout.save(layout_buf, format="JPEG", quality=95, dpi=(300, 300))
        layout_buf.seek(0)

        # Display cropped and layout images
        st.subheader("Cropped Passport Photo")
        st.image(img_cropped, caption="600×600 px, ready to upload", use_container_width=False)

        st.download_button(
            label="📥 Download Cropped Photo (JPEG)",
            data=cropped_buf,
            file_name="passport_photo.jpg",
            mime="image/jpeg"
        )

        st.subheader("Printable 6×4 Layout (4 Photos)")
        st.image(layout, caption="Printable 6×4 layout with borders", use_container_width=False)

        st.download_button(
            label="📥 Download 6×4 Layout (JPEG)",
            data=layout_buf,
            file_name="passport_layout.jpg",
            mime="image/jpeg"
        )

else:
    st.info("⬆️ Upload a photo above to start.")
