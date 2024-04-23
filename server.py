from flask import Flask, request, jsonify, render_template
import io
import torch
import numpy as np
from PIL import Image
import json
import os


app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.path.dirname(__file__), 'Mob2_torch.pth')
 # Update with the path to your model file in Google Colab
model = torch.load(model_path, map_location=device)

class_labels = {
    0: 'glioma_tumor',
    1: 'meningioma_tumor',
    2: 'no_tumor',
    3: 'pituitary_tumor'
}

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch tensor
    '''
    # Convert RGBA to RGB if the image has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Resize and crop
    image = image.resize((256, 256))  # Resize to fixed size
    image = image.crop((16, 16, 240, 240))  # Center crop 224x224

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np.array(image)/255 - mean) / std

    # PyTorch expects the color channel to be the first dimension
    image = image.transpose((2, 0, 1))

    # Convert to PyTorch tensor
    image = torch.tensor(image, dtype=torch.float32)

    # Add batch dimension
    image = image.unsqueeze(0)

    return image.to(device)

def predict(processed_image):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        output = model(processed_image)

    probabilities = torch.exp(output)

    # Convert probabilities and indices to lists
    probabilities = probabilities.squeeze().numpy()
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_labels[predicted_class_idx]
    predicted_prob = round(probabilities[predicted_class_idx] * 100, 2)  # Multiply by 100 here

    # Convert probabilities to percentages
    total_prob = sum(probabilities)
    predicted_confidence = round(predicted_prob / total_prob * 100, 2)

    probs_percent = {class_labels[i]: round(prob / total_prob * 100, 2) for i, prob in enumerate(probabilities)}

    return predicted_class, predicted_confidence, probs_percent

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']
    if img_file.filename == '':
        return jsonify({'error': 'No image provided'}), 400

    try:
        img = Image.open(io.BytesIO(img_file.read()))
    except IOError:
        return jsonify({'error': 'Invalid image'}), 400

    processed_image = process_image(img)
    predicted_class, predicted_prob, probs_percent = predict(processed_image)

    result = {
        'predicted_class': predicted_class,
        'predicted_prob': predicted_prob,
        'probs_percent': probs_percent
    }
    return jsonify(result), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

