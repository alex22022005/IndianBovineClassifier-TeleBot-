import os
import logging
import io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# --- Load Model ---
logger.info("Loading model for API...")
try:
    model = YOLO('best.pt')
    CLASS_NAMES = model.names
    logger.info("Model loaded successfully for API.")
except Exception as e:
    logger.error(f"A critical error occurred while loading the model: {e}")
    exit()

# --- Flask Web Server ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/api/classify', methods=['POST'])
def classify_image_endpoint():
    if 'image' not in request.files: return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    try:
        pil_image = Image.open(io.BytesIO(file.read()))
        results = model(pil_image)
        if results[0].probs is None: return jsonify({'error': 'No breed could be determined.'}), 404
        top1_prediction = results[0].probs.top1
        top1_confidence = results[0].probs.top1conf.item()
        breed_name = CLASS_NAMES[top1_prediction]
        return jsonify({'breedName': breed_name, 'confidence': f"{top1_confidence:.2%}"})
    except Exception as e:
        logger.error(f"Error during API image classification: {e}")
        return jsonify({'error': 'An internal error occurred during processing.'}), 500

# --- ADD THIS BLOCK AT THE END ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)