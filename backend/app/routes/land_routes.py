# app/routes/land_routes.py

from flask import Blueprint, request, jsonify

# Create a blueprint for land-related routes
bp = Blueprint('land', __name__)

@bp.route('/add', methods=['POST'])
def add_land():
    # Your logic for adding land data
    return jsonify({'message': 'Land data added successfully'})

@bp.route('/view', methods=['GET'])
def view_land():
    # Your logic for viewing land data
    return jsonify({'message': 'Land data fetched successfully'})
