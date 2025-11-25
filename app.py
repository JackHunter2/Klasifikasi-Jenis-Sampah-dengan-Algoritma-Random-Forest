from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
from PIL import Image
import joblib
from werkzeug.utils import secure_filename

from feature_extractor import extract_features_from_image_pil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Untuk flash messages
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Pastikan folder uploads ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load model & scaler ---
try:
    model = joblib.load('./model_files/waste_rf_model.pkl')
    scaler = joblib.load('./model_files/scaler.pkl')
    feature_cols = np.load('./model_files/feature_columns.npy', allow_pickle=True)
except FileNotFoundError as e:
    print(f"Error: File model tidak ditemukan - {e}")
    model = None
    scaler = None
    feature_cols = None


def allowed_file(filename):
    """Cek apakah ekstensi file diizinkan"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    filename = None
    error_message = None

    if request.method == 'POST':
        # Cek apakah model sudah dimuat
        if model is None or scaler is None:
            error_message = "Error: Model tidak dapat dimuat. Pastikan file model ada di folder model_files."
            return render_template('index.html',
                                 prediction=prediction,
                                 probability=probability,
                                 filename=filename,
                                 error_message=error_message)

        if 'file' not in request.files:
            error_message = "Tidak ada file yang dipilih."
            return render_template('index.html',
                                 prediction=prediction,
                                 probability=probability,
                                 filename=filename,
                                 error_message=error_message)
        
        file = request.files['file']

        if file.filename == '':
            error_message = "Tidak ada file yang dipilih."
            return render_template('index.html',
                                 prediction=prediction,
                                 probability=probability,
                                 filename=filename,
                                 error_message=error_message)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Load image
                img = Image.open(filepath).convert('RGB')

                # Extract features
                features = extract_features_from_image_pil(img)
                features = features.reshape(1, -1)

                # Scale
                features_scaled = scaler.transform(features)

                # Predict
                pred = model.predict(features_scaled)[0]
                prob = model.predict_proba(features_scaled)[0]

                label_map = {0: 'Organic (O)', 1: 'Recyclable (R)'}
                prediction = label_map[pred]
                probability = f"{np.max(prob)*100:.2f}%"
            except Exception as e:
                error_message = f"Error saat memproses gambar: {str(e)}"
                if filename and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                filename = None
        else:
            error_message = f"Format file tidak didukung. Gunakan: {', '.join(app.config['ALLOWED_EXTENSIONS'])}"

    return render_template('index.html',
                           prediction=prediction,
                           probability=probability,
                           filename=filename,
                           error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)
