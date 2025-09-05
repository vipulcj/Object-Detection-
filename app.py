from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import urllib.request
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__, template_folder='.')

# Configuration
app.config['SECRET_KEY'] = 'your_super_secret_key' 
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Model and Data Loading
try:
    # Load your trained model
    model = tf.keras.models.load_model('my_cnn_model.h5')
except (IOError, ImportError):
    model = None # Set model to None if it fails to load
    print("Warning: 'my_cnn_model.h5' not found or TensorFlow is not installed. The app will run without prediction capability.")


mean = 120.68732 
std = 64.14855

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Helper Functions
def allowed_file(filename):
    """Checks for allowed file extensions."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_source, is_url=False):
    """
    Loads an image from a URL or file path, and applies all necessary transformations.
    """
    try:
        if is_url:
            with urllib.request.urlopen(image_source) as resp:
                image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)
        else:
            image = cv2.imread(image_source, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError("Could not decode image. It may be corrupt or an unsupported format.")

        # Ensure image has 3 channels (e.g., handle grayscale or 4-channel PNGs)
        if len(image.shape) == 2: # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4: # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to the model's expected input size
        image = cv2.resize(image, (32, 32))
        
        # Normalize the image using the pre-calculated mean and std
        # Adding a small epsilon to std to prevent division by zero
        image = (image - mean) / (std + 1e-7)
        
        # Add an extra dimension for the batch
        image = image.reshape((1, 32, 32, 3))
        
        return image
    except Exception as e:
        # Re-raise exceptions to be caught by the main prediction route
        raise ValueError(f"Image processing failed: {e}")


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles both file uploads and URL submissions for prediction."""
    if model is None:
        return jsonify({'error': 'Model is not loaded. Please check the server logs.'}), 500

    try:
        image_to_predict = None
        
        # Case 1: Prediction from a URL
        if request.is_json:
            data = request.get_json()
            if 'url' not in data or not data['url']:
                return jsonify({'error': 'No URL provided in the request.'}), 400
            
            url = data['url']
            image_to_predict = process_image(url, is_url=True)

        # Case 2: Prediction from a file upload
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected.'}), 400

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_to_predict = process_image(filepath)
            else:
                return jsonify({'error': 'File type not allowed.'}), 400
        else:
             return jsonify({'error': 'Invalid request. Please provide a file or a URL.'}), 400

        # Make the prediction
        predictions = model.predict(image_to_predict)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        
        return jsonify({'prediction': predicted_class_name})

    except Exception as e:
        # Return any errors that occurred during processing or prediction
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For production, use a proper WSGI server like Gunicorn.
    app.run(debug=True)
