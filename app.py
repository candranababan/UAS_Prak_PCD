from flask import Flask, render_template, request
import cv2
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Batas upload aman
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def resize_image(img, max_size=700):
    h, w, _ = img.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img


def process_image(img, method, thresh):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.uint8)

    if method == "rgb":
        return rgb
    elif method == "hsv":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    elif method == "threshold":
        _, r = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        return r
    elif method == "otsu":
        _, r = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return r
    elif method == "erosi":
        return cv2.erode(gray, kernel, 1)
    elif method == "dilasi":
        return cv2.dilate(gray, kernel, 1)
    elif method == "opening":
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif method == "closing":
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    elif method == "floodfill":
        _, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        h, w = th.shape
        mask = np.zeros((h+2, w+2), np.uint8)
        flood = th.copy()
        cv2.floodFill(flood, mask, (0, 0), 255)
        return th | cv2.bitwise_not(flood)

    return gray


@app.route("/", methods=["GET", "POST"])
def index():
    original = None
    result = None
    method = "rgb"
    thresh = 120

    if request.method == "POST":
        method = request.form.get("method")
        thresh = int(request.form.get("threshold", 120))

        file = request.files.get("image")

        # Upload baru (boleh kosong)
        if file and file.filename != "":
            img = Image.open(file).convert("RGB")
            img_np = resize_image(np.array(img))

            original_path = os.path.join(UPLOAD_FOLDER, "original.png")
            cv2.imwrite(original_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        # Jika gambar sudah ada → proses
        original_path = os.path.join(UPLOAD_FOLDER, "original.png")
        if os.path.exists(original_path):
            img_np = cv2.imread(original_path)

            processed = process_image(img_np, method, thresh)
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

            result_path = os.path.join(UPLOAD_FOLDER, "result.png")
            cv2.imwrite(result_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

            original = "uploads/original.png"
            result = "uploads/result.png"

    return render_template(
        "index.html",
        original=original,
        result=result,
        method=method,
        thresh=thresh
    )


if __name__ == "__main__":
    app.run(debug=True)
