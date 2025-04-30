import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'default_secret_key'
    MONGO_URI = os.environ.get('MONGO_URI')  # Get Mongo URI from environment variables
