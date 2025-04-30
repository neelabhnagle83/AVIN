from flask import Flask
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

# Initialize JWTManager
jwt = JWTManager(app)

# Registering Blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(disease_bp, url_prefix='/disease')
app.register_blueprint(land_bp, url_prefix='/land')
app.register_blueprint(store_bp, url_prefix='/store')
app.register_blueprint(chatbot_bp, url_prefix='/chatbot')

if __name__ == '__main__':
    app.run(debug=True)
