# app/routes/disease_routes.py

from flask import Blueprint, request, jsonify

# Create a blueprint for disease-related routes
bp = Blueprint('disease', __name__)

@bp.route('/upload', methods=['POST'])
def upload_image():
    # Your disease detection logic goes here
    return jsonify({'message': 'Image uploaded successfully for disease detection'})


@bp.route('/scan', methods=['POST'])
def scan_disease():
    # Your disease scanning logic goes here
    return jsonify({'message': 'Scanning image for disease...'})
