import streamlit as st
from PIL import Image, ImageOps
import io

st.set_page_config(page_title="Passport Photo Cropper", page_icon="📸", layout="centered")

st.title("📸 Passport Photo Cropper & Layout Generator")
st.markdown("""
Upload a photo, crop it, and generate passport-ready images + a 6×4 printable layout (4 copies).
""")

# --- Cache image loading ---
@st.cache_data
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    # Prevent memory issues — shrink very large images
    if max(img.size) > 2000:
        ratio = 2000 / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return img

def compress_image(img, min_kb=20, max_kb=100):
    quality = 90
    buf = io.BytesIO()
    for _ in range(10):
        buf.seek(0)
        img.save(buf, format="JPEG", quality=quality)
        size_kb = buf.tell() / 1024
        if min_kb <= size_kb <= max_kb:
            break
        quality -= 5 if size_kb > max_kb else -5
    buf.seek(0)
    return buf

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = load_image(uploaded_file)
    st.image(img, caption=f"Uploaded image ({img.width}×{img.height}px)", use_container_width=True)

    st.subheader("Adjust Crop Area")
    st.markdown("Use the sliders to crop your photo manually.")

    # Crop sliders
    left = st.slider("Left", 0, img.width - 1, 0)
    right = st.slider("Right", left + 1, img.width, img.width)
    top = st.slider("Top", 0, img.height - 1, 0)
    bottom = st.slider("Bottom", top + 1, img.height, img.height)

    if st.button("✂️ Crop & Generate"):
        # Safety: ensure valid box
        if left < right and top < bottom:
            cropped = img.crop((left, top, right, bottom))
            cropped = cropped.resize((600, 600), Image.LANCZOS)
            cropped.info["dpi"] = (300, 300)
            cropped_buf = compress_image(cropped)

            # --- Create printable 6×4 layout (4 copies) ---
            layout = Image.new("RGB", (1800, 1200), "white")  # 6×4" @ 300 DPI
            bordered = ImageOps.expand(cropped, border=10, fill="lightgray")
            margin_x, margin_y = 150, 100
            for row in range(2):
                for col in range(2):
                    x = margin_x + col * (bordered.width + margin_x)
                    y = margin_y + row * (bordered.height + margin_y)
                    layout.paste(bordered, (x, y))

            layout_buf = io.BytesIO()
            layout.save(layout_buf, format="JPEG", quality=95, dpi=(300, 300))
            layout_buf.seek(0)

            # Display + download buttons
            st.subheader("✅ Cropped Passport Photo (600×600)")
            st.image(cropped, use_container_width=False)
            st.download_button("📥 Download Cropped Photo", cropped_buf, "passport_photo.jpg", "image/jpeg")

            st.subheader("🖨️ Printable 6×4 Layout (4 Photos)")
            st.image(layout, use_container_width=False)
            st.download_button("📥 Download 6×4 Layout", layout_buf, "passport_layout.jpg", "image/jpeg")
        else:
            st.error("Invalid crop area — make sure Right > Left and Bottom > Top.")
else:
    st.info("⬆️ Upload a photo above to start.")
