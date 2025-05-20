# app/routes/disease_routes.py

from flask import Blueprint, request, jsonify
from app.services.disease_service import process_images

# Create a blueprint for disease-related routes
bp = Blueprint('disease', __name__)

@bp.route('/upload', methods=['POST'])
def upload_image():
    return jsonify({'message': 'Image uploaded successfully for disease detection'})

@bp.route('/scan', methods=['POST'])
def scan_disease():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    images = request.files.getlist('images')
    try:
        result = process_images(images)
        return jsonify(result)  # Return the full result object including disease and confidence
    except Exception as e:
        return jsonify({'error': str(e)}), 500
