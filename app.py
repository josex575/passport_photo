from flask import Flask, render_template_string, request, send_file
from PIL import Image, ImageOps, ImageDraw
import io
import os

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Passport Photo Cropper</title>
</head>
<body style="font-family: sans-serif; text-align:center; padding-top:30px;">
    <h2>Passport Photo Crop Tool</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="photo" accept="image/*" required>
        <br><br>
        <button type="submit">Upload & Crop</button>
    </form>
    {% if cropped_url %}
        <h3>Cropped Passport Photo:</h3>
        <img src="{{ cropped_url }}" alt="Cropped" style="border:1px solid #ccc; width:200px;">
        <br><a href="{{ cropped_url }}" download="passport_photo.jpg">Download Cropped Photo</a>
        <br><br>
        <h3>Printable 6x4 Layout:</h3>
        <img src="{{ layout_url }}" alt="Layout" style="border:1px solid #ccc; width:300px;">
        <br><a href="{{ layout_url }}" download="passport_layout.jpg">Download 6x4 Layout</a>
    {% endif %}
</body>
</html>
"""

def compress_image(img, min_kb=20, max_kb=100):
    """Compress image to fit file size constraints"""
    quality = 95
    buf = io.BytesIO()
    while True:
        buf.seek(0)
        img.save(buf, format="JPEG", quality=quality)
        size_kb = buf.tell() / 1024
        if size_kb <= max_kb and size_kb >= min_kb:
            break
        if size_kb > max_kb:
            quality -= 5
        elif size_kb < min_kb:
            quality += 5
        if quality < 10 or quality > 95:
            break
    buf.seek(0)
    return buf

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["photo"]
        img = Image.open(file).convert("RGB")

        # Crop to square (center)
        side = min(img.size)
        left = (img.width - side) / 2
        top = (img.height - side) / 2
        right = (img.width + side) / 2
        bottom = (img.height + side) / 2
        img = img.crop((left, top, right, bottom))

        # Resize to passport specs (e.g., 600x600)
        img = img.resize((600, 600), Image.LANCZOS)

        # Ensure bit depth and DPI
        img.info['dpi'] = (300, 300)

        # Compress to meet KB limit
        cropped_buf = compress_image(img)

        # Make layout image (6x4 inch at 300 DPI → 1800x1200 px)
        layout = Image.new("RGB", (1800, 1200), "white")
        bordered = ImageOps.expand(img, border=10, fill='lightgray')

        # Center the image
        x = (layout.width - bordered.width) // 2
        y = (layout.height - bordered.height) // 2
        layout.paste(bordered, (x, y))

        layout_buf = io.BytesIO()
        layout.save(layout_buf, format="JPEG", quality=95, dpi=(300, 300))
        layout_buf.seek(0)

        # Save temporarily to memory
        cropped_path = "static/cropped.jpg"
        layout_path = "static/layout.jpg"
        os.makedirs("static", exist_ok=True)
        with open(cropped_path, "wb") as f:
            f.write(cropped_buf.getbuffer())
        with open(layout_path, "wb") as f:
            f.write(layout_buf.getbuffer())

        return render_template_string(HTML, cropped_url="/static/cropped.jpg", layout_url="/static/layout.jpg")

    return render_template_string(HTML)

if __name__ == "__main__":
    app.run(debug=True)
