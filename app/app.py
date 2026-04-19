from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ── Symptom model (unchanged) ──────────────────────────
model  = joblib.load('../models/model.pkl')
scaler = joblib.load('../models/scaler.pkl')

# ── Rebuild CNN architecture (same as Colab Cell 6) ────
def build_cnn_model():
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights=None        # no imagenet weights needed here
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(128, 128, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# ── Load CNN with weights 
cnn_model = build_cnn_model()
cnn_model.load_weights('../models/cnn_weights.weights.h5')
# cnn_model = load_model('../models/cnn_weights.weights.h5', compile=False)
print("CNN model loaded successfully")

CNN_CLASSES = ['Benign', 'Malignant', 'Normal']
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ── Home page 
@app.route('/')
def home():
    return render_template('index.html')

# ── Symptom prediction (unchanged) 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            int(request.form['gender']),
            int(request.form['age']),
            int(request.form['smoking']),
            int(request.form['yellow_fingers']),
            int(request.form['anxiety']),
            int(request.form['peer_pressure']),
            int(request.form['chronic_disease']),
            int(request.form['fatigue']),
            int(request.form['allergy']),
            int(request.form['wheezing']),
            int(request.form['alcohol']),
            int(request.form['coughing']),
            int(request.form['shortness_breath']),
            int(request.form['swallowing_diff']),
            int(request.form['chest_pain']),
        ]

        features_scaled = scaler.transform([features])
        prediction      = model.predict(features_scaled)[0]
        probability     = model.predict_proba(features_scaled)[0][1] * 100

        risk   = 'high' if prediction == 1 else 'low'
        result = 'HIGH RISK' if prediction == 1 else 'LOW RISK'

    except Exception as e:
        result      = f'Error: {str(e)}'
        probability = 0
        risk        = 'low'

    return render_template('index.html',
                           result=result,
                           probability=round(probability, 1),
                           risk=risk)

# ── CNN image prediction (new) ──────────────────────────
@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'ct_scan' not in request.files:
            return render_template('index.html',
                                   cnn_result='Error: No image uploaded',
                                   cnn_risk='low',
                                   cnn_confidence=0)

        file = request.files['ct_scan']

        if file.filename == '':
            return render_template('index.html',
                                   cnn_result='Error: No image selected',
                                   cnn_risk='low',
                                   cnn_confidence=0)

        # Save uploaded image temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction      = cnn_model.predict(img_array)
        predicted_idx   = np.argmax(prediction)
        predicted_class = CNN_CLASSES[predicted_idx]
        confidence      = float(np.max(prediction)) * 100

        # Delete temp image
        os.remove(filepath)

        # Set risk level
        if predicted_class == 'Malignant':
            cnn_risk   = 'high'
            cnn_result = 'MALIGNANT — HIGH RISK'
        elif predicted_class == 'Benign':
            cnn_risk   = 'medium'
            cnn_result = 'BENIGN — LOW RISK'
        else:
            cnn_risk   = 'low'
            cnn_result = 'NORMAL — NO CANCER DETECTED'

        cnn_probs = {
            'Benign'   : round(float(prediction[0][0]) * 100, 1),
            'Malignant': round(float(prediction[0][1]) * 100, 1),
            'Normal'   : round(float(prediction[0][2]) * 100, 1),
        }

    except Exception as e:
        cnn_result  = f'Error: {str(e)}'
        cnn_risk    = 'low'
        confidence  = 0
        cnn_probs   = {'Benign': 0, 'Malignant': 0, 'Normal': 0}

    return render_template('index.html',
                           cnn_result=cnn_result,
                           cnn_risk=cnn_risk,
                           cnn_confidence=round(confidence, 1),
                           cnn_probs=cnn_probs)

if __name__ == '__main__':
    app.run(debug=True)