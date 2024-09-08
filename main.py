from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import os

# Initialize Flask app
app = Flask(__name__)

# Define the custom loss function (replace this with the actual function)
@tf.keras.utils.register_keras_serializable()
def weighted_loss(y_true, y_pred):
    # Replace with the actual weighted loss function
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Define the paths to your model files
MODEL_FOLDER = 'models'
# Load models with custom loss function
model_chest = load_model(os.path.join(MODEL_FOLDER, 'chestXray_model.keras'), custom_objects={'weighted_loss': weighted_loss})
model_fracture = load_model(os.path.join(MODEL_FOLDER, 'fracture.keras'), custom_objects={'weighted_loss': weighted_loss})
model_brain = load_model(os.path.join(MODEL_FOLDER, 'brain_tumor_dataset.keras'), custom_objects={'weighted_loss': weighted_loss})

# Helper functions
def load_image(path, target_size):
    img = keras_image.load_img(path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict_fracture', methods=['POST'])
def predict_fracture():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        img_array = load_image(file_path, (224, 224))
        prediction = model_fracture.predict(img_array)
        probability = prediction[0][0]
        class_label = 'The bone is fractured' if probability >= 0.2 else 'The bone is not fractured'
        return jsonify({'class_label': class_label, 'probability': float(probability)})

@app.route('/predict_chest', methods=['POST'])
def predict_chest():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        img_array = load_image(file_path, (320, 320))
        predictions = model_chest.predict(img_array)
        predictions = predictions[0].tolist()  # Convert to list
        labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
                  'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
        results = dict(zip(labels, predictions))
        return jsonify(results)

@app.route('/predict_tumor', methods=['POST'])
def predict_tumor():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        # Correct image loading for TensorFlow
        img = keras_image.load_img(file_path, target_size=(256, 256))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model_brain.predict(img_array)
        probability = prediction[0][0]  # Get the probability value
        result = "The person has Brain Tumor" if probability > 0.5 else "The Person Does not have Brain Tumor"
        return jsonify({'result': result, 'probability': float(probability)})

# Run the server
if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)