from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import librosa
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the pre-trained model
model = load_model('your_model_path.h5')  # Replace 'your_model_path.h5' with the actual path to your model file

# Audio Feature Extraction
def extract_audio_features(filename, subfolder):
    audio_path = f'train_audio/{subfolder}/{filename}'
    x, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
    return mfccs.flatten()

# Text Data Handling
def preprocess_text(description):
    vectorizer = CountVectorizer()
    text_features = vectorizer.fit_transform([description]).toarray()
    return text_features

# Image Preprocessing
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        image_file = request.files['image']
        audio_file = request.files['audio']
        description = request.form['description']

        # Save the image and audio files
        image_path = 'uploads/image.jpg'
        audio_path = 'uploads/audio.wav'
        image_file.save(image_path)
        audio_file.save(audio_path)

        # Extract features
        image_features = load_and_preprocess_image(image_path)
        audio_features = extract_audio_features('audio.wav', 'subfolder')  # Replace 'subfolder' with the actual subfolder
        text_features = preprocess_text(description)

        # Make predictions
        combined_features = [image_features, audio_features, text_features]
        prediction = model.predict(combined_features)
        prediction = prediction[0][0]  # Assuming it's a binary classification task

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
