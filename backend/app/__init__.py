import logging
from flask import Flask
from flask_pymongo import PyMongo

mongo = PyMongo()  # Initialize PyMongo instance

def create_app():
    app = Flask(__name__)
    app.config["MONGO_URI"] = "mongodb://localhost:27017/avin_db"  # MongoDB URI
    mongo.init_app(app)

    # Debug logging for MongoDB connection
    try:
        mongo.db.command('ping')  # Ping the database to ensure connection
        app.logger.info("Successfully connected to MongoDB.")
    except Exception as e:
        app.logger.error(f"MongoDB connection error: {e}")

    return app
