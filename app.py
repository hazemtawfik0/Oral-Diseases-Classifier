import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, 'oral_disease_resnet50.h5')

print("Loading model... Please wait. This might take a few seconds.")
model = load_model(MODEL_PATH)
print("Model loaded successfully! Ready for predictions.")

class_names = ['Calculus', 'Data caries', 'Gingivitis', 'Mouth Ulcer', 'Tooth Discoloration', 'hypodontia']

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    return {
        'class': class_names[predicted_class_idx],
        'confidence': confidence
    }

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        img_bytes = file.read()
        result = predict_image(img_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oral Diseases Classifier</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; display: flex; flex-direction: column;
            align-items: center; padding: 40px 20px; color: #333;
        }
        .container {
            background: white; border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 50px; max-width: 900px; width: 100%;
        }
        h1 { text-align: center; color: #667eea; margin-bottom: 10px; font-size: 2.5em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 40px; font-size: 1.1em; }
        .upload-form { text-align: center; margin-bottom: 40px; }
        .file-input-wrapper { margin: 30px 0; position: relative; }
        .file-input {
            opacity: 0; width: 100%; height: 100%; position: absolute;
            cursor: pointer; top: 0; left: 0;
        }
        .file-label {
            display: inline-block; padding: 15px 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border-radius: 50px; font-size: 1.1em; cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .file-label:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6); }
        .submit-btn {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white; border: none; padding: 15px 50px; font-size: 1.2em;
            cursor: pointer; border-radius: 50px; transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
        }
        .submit-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(17, 153, 142, 0.6); }
        .submit-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .result {
            padding: 40px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px; text-align: center;
        }
        .result h2 { color: #667eea; margin-bottom: 20px; font-size: 1.8em; }
        .confidence { font-size: 2.5em; font-weight: bold; color: #11998e; margin-top: 10px; }
        .preview {
            max-width: 100%; height: auto; margin: 20px 0;
            border-radius: 10px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        .predicted-class { font-size: 1.5em; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Oral Diseases Image Classifier</h1>
        <p class="subtitle">Upload an image to get a prediction</p>
        <div class="upload-form">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="file-input-wrapper">
                    <input type="file" id="imageInput" name="file" accept="image/*" class="file-input" required>
                    <label for="imageInput" class="file-label">Choose Image</label>
                </div>
                <br>
                <button type="submit" id="submitBtn" class="submit-btn">Classify Image</button>
            </form>
        </div>
        <div id="result" class="result" style="display: none;">
            <h2>Classification Result</h2>
            <img id="preview" class="preview">
            <p class="predicted-class">Predicted Class: <strong id="predictedClass"></strong></p>
            <p>Confidence: <span id="confidence" class="confidence"></span></p>
        </div>
    </div>
    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const fileInput = document.getElementById('imageInput');
            const submitBtn = document.getElementById('submitBtn');
            const file = fileInput.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            try {
                submitBtn.disabled = true;
                submitBtn.textContent = 'Classifying...';
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                if(result.error) {
                    alert('Error: ' + result.error);
                } else {
                    document.getElementById('preview').src = URL.createObjectURL(file);
                    document.getElementById('predictedClass').textContent = result['class'];
                    document.getElementById('confidence').textContent = (result['confidence'] * 100).toFixed(2) + '%';
                    document.getElementById('result').style.display = 'block';
                }
            } catch (error) {
                alert('Error classifying image: ' + error);
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Classify Image';
            }
        });
    </script>
</body>
</html>
    '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)