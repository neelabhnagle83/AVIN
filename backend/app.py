from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from model import predict_disease
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from app.routes.auth_routes import bp as auth_bp
from app.routes.disease_routes import bp as disease_bp
from app.routes.land_routes import bp as land_bp
from app.routes.store_routes import bp as store_bp
from app.routes.chatbot_routes import bp as chatbot_bp
from app import create_app

# Initialize app using the factory method
app = create_app()

CORS(app)  # Enable CORS to allow cross-origin requests

# Configurations
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/avin_db'  # MongoDB URI
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'  # Add JWT secret key

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize JWTManager
jwt = JWTManager(app)

# Registering Blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(disease_bp, url_prefix='/disease')
app.register_blueprint(land_bp, url_prefix='/land')
app.register_blueprint(store_bp, url_prefix='/store')
app.register_blueprint(chatbot_bp, url_prefix='/chatbot')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Predict disease
    result = predict_disease(filepath)

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # Updated to bind the server to all network interfaces
