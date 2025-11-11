from flask import Flask, render_template_string, request
from PIL import Image, ImageOps
import io, os, base64

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Passport Photo Crop Tool</title>
    <link  href="https://unpkg.com/cropperjs/dist/cropper.min.css" rel="stylesheet"/>
    <script src="https://unpkg.com/cropperjs"></script>
    <style>
        body { font-family: sans-serif; text-align: center; padding: 20px; }
        img { max-width: 100%; }
        .container { display: inline-block; max-width: 500px; margin: 20px auto; }
        button { margin-top: 10px; padding: 8px 16px; font-size: 16px; }
    </style>
</head>
<body>
    <h2>Passport Photo Cropper</h2>
    <p>Upload an image, crop to your face, and generate printable passport photos (4 per 6×4 sheet).</p>

    <input type="file" id="upload" accept="image/*">
    <div class="container">
        <img id="image" style="max-width:100%; display:none;">
    </div>
    <br>
    <button id="cropBtn" style="display:none;">Crop & Upload</button>

    {% if cropped_url %}
        <h3>Cropped Passport Photo:</h3>
        <img src="{{ cropped_url }}" style="border:1px solid #ccc; width:200px;"><br>
        <a href="{{ cropped_url }}" download="passport_photo.jpg">Download Cropped Photo</a>
        <h3>Printable 6x4 Layout (4 copies):</h3>
        <img src="{{ layout_url }}" style="border:1px solid #ccc; width:300px;"><br>
        <a href="{{ layout_url }}" download="passport_layout.jpg">Download 6x4 Layout</a>
    {% endif %}

    <form id="cropForm" method="POST" enctype="multipart/form-data" style="display:none;">
        <input type="hidden" name="cropped_data" id="cropped_data">
    </form>

<script>
let cropper;
const upload = document.getElementById('upload');
const image = document.getElementById('image');
const cropBtn = document.getElementById('cropBtn');
const form = document.getElementById('cropForm');
const croppedInput = document.getElementById('cropped_data');

upload.addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(event) {
        image.src = event.target.result;
        image.style.display = 'block';
        if (cropper) cropper.destroy();
        cropper = new Cropper(image, {
            aspectRatio: 1,
            viewMode: 1,
            dragMode: 'move',
            autoCropArea: 1.0,
        });
        cropBtn.style.display = 'inline-block';
    };
    reader.readAsDataURL(file);
});

cropBtn.addEventListener('click', () => {
    if (!cropper) return;
    const canvas = cropper.getCroppedCanvas({ width: 600, height: 600 });
    canvas.toBlob(blob => {
        const reader = new FileReader();
        reader.onloadend = () => {
            croppedInput.value = reader.result;
            form.submit();
        };
        reader.readAsDataURL(blob);
    }, 'image/jpeg', 0.9);
});
</script>
</body>
</html>
"""

def compress_image(img, min_kb=20, max_kb=100):
    """Compress image to fit file size limits"""
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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data_url = request.form.get("cropped_data")
        if not data_url:
            return render_template_string(HTML)

        # Decode base64 image from frontend
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Resize & set DPI
        img = img.resize((600, 600), Image.LANCZOS)
        img.info["dpi"] = (300, 300)

        # Compress cropped image
        cropped_buf = compress_image(img)

        # --- Create 6x4 layout with 4 copies ---
        layout = Image.new("RGB", (1800, 1200), "white")  # 6x4" @ 300dpi
        bordered = ImageOps.expand(img, border=10, fill="lightgray")

        # 2 columns x 2 rows (4 copies total)
        margin_x, margin_y = 150, 100  # spacing
        for row in range(2):
            for col in range(2):
                x = margin_x + col * (bordered.width + margin_x)
                y = margin_y + row * (bordered.height + margin_y)
                layout.paste(bordered, (x, y))

        layout_buf = io.BytesIO()
        layout.save(layout_buf, format="JPEG", quality=95, dpi=(300, 300))
        layout_buf.seek(0)

        # Save output files
        os.makedirs("static", exist_ok=True)
        cropped_path = "static/cropped.jpg"
        layout_path = "static/layout.jpg"
        with open(cropped_path, "wb") as f:
            f.write(cropped_buf.getbuffer())
        with open(layout_path, "wb") as f:
            f.write(layout_buf.getbuffer())

        return render_template_string(HTML, cropped_url="/static/cropped.jpg", layout_url="/static/layout.jpg")

    return render_template_string(HTML)

if __name__ == "__main__":
    # ✅ Safe for Python 3.13 — no reloader
    app.run(debug=True, use_reloader=False)
