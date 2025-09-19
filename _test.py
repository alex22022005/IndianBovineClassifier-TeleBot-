import os
import logging
import json
import google.generativeai as genai
from telegram import Update, ParseMode
from telegram.error import TimedOut
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

# --- Setup ---

# --- Flask Web Server ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

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

def run_flask_app():
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask server. Access the website at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

# --- Main Execution Block ---


if __name__ == '__main__':
    main()