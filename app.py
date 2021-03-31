from flask import Flask, jsonify, request, flash, redirect, url_for
from sayac import test
from werkzeug.utils import secure_filename
import os
import uuid

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def baslangic_api():
    result = test.sayacdondur("images/16.jpg")
    return jsonify({"Sonuc": result})


@app.route('/sayac', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # filename=str(uuid.uuid4())
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = test.sayacdondur(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify({"Sonuc": result})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# if __name__ == "__main__":
#     app.run()
