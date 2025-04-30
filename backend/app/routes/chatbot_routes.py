# app/routes/chatbot_routes.py

from flask import Blueprint, request, jsonify

# Create a blueprint for chatbot-related routes
bp = Blueprint('chatbot', __name__)

@bp.route('/ask', methods=['POST'])
def ask_chatbot():
    # Your logic for the chatbot
    return jsonify({'message': 'Chatbot response'})


@bp.route('/train', methods=['POST'])
def train_chatbot():
    # Your logic for training the chatbot
    return jsonify({'message': 'Training chatbot'})
