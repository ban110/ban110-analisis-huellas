from fingerprint import generate_from_image
from flask import Flask, flash, request, redirect, jsonify
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/received'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'BMP'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=["POST"])
def analyze_image():
    saved_image_path = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            saved_image_path = "./{}/{}".format(UPLOAD_FOLDER, filename)
            file.save(saved_image_path)
            encode, prom = generate_from_image(saved_image_path)
            return jsonify(encode=str(encode), distance=prom)
    return "hello"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
