# app/routes/store_routes.py

from flask import Blueprint, request, jsonify

# Create a blueprint for store-related routes
bp = Blueprint('store', __name__)

@bp.route('/add', methods=['POST'])
def add_product():
    # Your logic for adding a product to the store
    return jsonify({'message': 'Product added successfully'})

@bp.route('/view', methods=['GET'])
def view_products():
    # Your logic for viewing products from the store
    return jsonify({'message': 'Store products fetched successfully'})
