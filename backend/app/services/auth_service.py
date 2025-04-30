from flask_jwt_extended import create_access_token
import re

def generate_access_token(identity):
    return create_access_token(identity=identity)

# =====================
# Input Validation Functions
# =====================

def validate_email(email):
    # Email must end with '@gmail.com'
    return email.endswith('@gmail.com')

def validate_password(password):
    # Password must have at least 6 characters, contain at least 1 letter and 1 number
    pattern = r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{6,}$'
    return bool(re.match(pattern, password))

def validate_phone(phone):
    # Phone must be exactly 10 digits
    return phone.isdigit() and len(phone) == 10
