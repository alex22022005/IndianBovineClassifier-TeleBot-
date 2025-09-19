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
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)

# --- Load Models and Data ---
logger.info("Loading models and data...")
try:
    model = YOLO('best.pt')
    CLASS_NAMES = model.names
    with open('breed_info.json', 'r', encoding='utf-8') as f:
        breed_descriptions = json.load(f)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-pro')
    MONGO_URI = os.getenv('MONGO_URI')
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000)
    db = client.get_database('Bovine_Classifier')
    user_interactions_collection = db.get_collection('user_interactions')
    logger.info("All models and data loaded successfully.")
except Exception as e:
    logger.error(f"A critical error occurred during initialization: {e}")
    exit()

# --- Telegram Bot Handlers ---
def start(update: Update, context: CallbackContext) -> None:
    user_name = update.effective_user.first_name
    welcome_message = f"👋 Hello, {user_name}!\n\nI am your Cattle & Buffalo Breed Identification Bot. 🐄🐃"
    update.message.reply_text(welcome_message)

def help_command(update: Update, context: CallbackContext) -> None:
    help_text = "**How to use this bot:**\n1. **Identification:** Send an image.\n2. **Follow-up Questions:** Ask me anything after an identification."
    update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

def handle_image(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    photo_path = None
    try:
        context.bot.send_message(chat_id, "Downloading your image...")
        photo_file = update.message.photo[-1].get_file(timeout=20)
        photo_path = f'{photo_file.file_id}.jpg'
        photo_file.download(photo_path)
    except TimedOut:
        logger.warning("Timeout while downloading image from Telegram.")
        update.message.reply_text("I had trouble downloading your image due to a network issue. Could you please try sending it again?")
        return
    except Exception as e:
        logger.error(f"An unknown error occurred during image download: {e}")
        update.message.reply_text("An unexpected error occurred. Please try again.")
        return

    try:
        context.bot.send_message(chat_id, "Analyzing your image... 🧐")
        pil_image = Image.open(photo_path)
        results = model(pil_image)
        if results[0].probs is None:
            update.message.reply_text("I'm sorry, I couldn't determine the breed from this image.")
            return

        top1_prediction = results[0].probs.top1
        top1_confidence = results[0].probs.top1conf.item()
        breed_name = CLASS_NAMES[top1_prediction]
        confidence = top1_confidence
        
        logger.info(f"Telegram User {update.effective_user.id} detected {breed_name} with confidence {confidence:.2f}")
        context.user_data['last_identified_breed'] = breed_name
        
        user = update.effective_user
        interaction_id = datetime.utcnow().isoformat()
        new_interaction = {
            "interaction_id": interaction_id, "timestamp": datetime.utcnow(),
            "identified_breed": breed_name, "confidence": confidence, "queries": []
        }
        user_interactions_collection.update_one(
            {"user_id": user.id},
            {"$set": {"username": user.username, "first_name": user.first_name}, "$push": {"interactions": new_interaction}},
            upsert=True
        )
        context.user_data['last_interaction_id'] = interaction_id
        description = breed_descriptions.get(breed_name, {}).get('features', "No description available.")
        response_text = (f"**Breed Identified:** {breed_name}\n**Confidence:** {confidence:.2%}\n\n"
                         f"**Description:**\n{description}\n\nFeel free to ask me anything about this breed!")
        update.message.reply_photo(photo=open(photo_path, 'rb'), caption=response_text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Error during bot image processing: {e}")
        update.message.reply_text("An error occurred while analyzing the image.")
    finally:
        if photo_path and os.path.exists(photo_path):
            os.remove(photo_path)

def handle_text(update: Update, context: CallbackContext) -> None:
    user_question = update.message.text
    user = update.effective_user
    chat_id = update.effective_chat.id
    
    thinking_message = context.bot.send_message(chat_id=chat_id, text="Thinking Deeply for best Response... 🤔")
    gemini_done_event = threading.Event()

    def send_interim_message():
        if not gemini_done_event.wait(timeout=10):
            try:
                context.bot.edit_message_text(chat_id=chat_id, message_id=thinking_message.message_id, text="Almost there...")
            except Exception as e:
                logger.warning(f"Could not edit interim message: {e}")

    interim_thread = threading.Thread(target=send_interim_message)
    interim_thread.start()

    last_breed = context.user_data.get('last_identified_breed')
    last_interaction_id = context.user_data.get('last_interaction_id')
    
    if last_breed:
        prompt = (f"You are a helpful Indian livestock expert. A user has identified a '{last_breed}' breed. "
                  f"Answer their question concisely. User's Question: '{user_question}'")
    else:
        prompt = (f"You are a helpful Indian livestock expert. Answer the user's general question about "
                  f"Indian cattle or buffaloes concisely. User's Question: '{user_question}'")

    try:
        response = gemini_model.generate_content(prompt)
        bot_answer = response.text
        gemini_done_event.set()
        
        context.bot.edit_message_text(chat_id=chat_id, message_id=thinking_message.message_id, text=bot_answer)

        query_log = {"question": user_question, "answer": bot_answer, "timestamp": datetime.utcnow()}
        if last_breed and last_interaction_id:
            user_interactions_collection.update_one(
                {"user_id": user.id, "interactions.interaction_id": last_interaction_id},
                {"$push": {"interactions.$.queries": query_log}}
            )
        else:
            general_interaction = {
                "interaction_id": datetime.utcnow().isoformat(), "timestamp": datetime.utcnow(),
                "identified_breed": "General Question", "confidence": None, "queries": [query_log]
            }
            user_interactions_collection.update_one(
                {"user_id": user.id},
                {"$set": {"username": user.username, "first_name": user.first_name}, "$push": {"interactions": general_interaction}},
                upsert=True
            )
        logger.info(f"Logged new query for Telegram user {user.id}")
    except Exception as e:
        gemini_done_event.set()
        logger.error(f"Error calling Gemini or logging text for bot: {e}")
        context.bot.edit_message_text(chat_id=chat_id, message_id=thinking_message.message_id, text="I'm having trouble connecting to my knowledge base. Please try again later.")

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
def main() -> None:
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()
    
    TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables.")
        return
        
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.photo, handle_image))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))
    
    updater.start_polling()
    logger.info("Telegram Bot is polling...")
    updater.idle()

if __name__ == '__main__':
    main()