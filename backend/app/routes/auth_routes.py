from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from app.models.user_model import UserModel
from app import mongo
from app.services.auth_service import validate_email, validate_password, validate_phone, generate_access_token
from bson import ObjectId
from werkzeug.security import generate_password_hash
import re

bp = Blueprint('auth', __name__)

# ========================
# Register (Signup)
# ========================
@bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    name = data.get('name')
    phone = data.get('phone')
    password = data.get('password')

    # Input validations
    if not validate_email(email):
        return jsonify({"message": "Invalid email. Email must end with @gmail.com"}), 400
    if not validate_phone(phone):
        return jsonify({"message": "Invalid phone number. It must be exactly 10 digits"}), 400
    if not validate_password(password):
        return jsonify({"message": "Weak password. It must contain letters, numbers, and be at least 6 characters."}), 400

    # Check if email already exists
    if UserModel.find_by_email(email):
        return jsonify({"message": "Email already exists"}), 400

    # Check if phone number already exists
    if UserModel.find_by_phone(phone):
        return jsonify({"message": "Phone number already exists"}), 400

    # Create the new user in the database
    UserModel.create_user(email, name, phone, password)

    return jsonify({"message": "User registered successfully"}), 201

# ========================
# Login with Email & Password
# ========================
@bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = UserModel.find_by_email(email)

    if not user or not UserModel.verify_password(user['password'], password):
        return jsonify({"message": "Invalid credentials"}), 401

    access_token = generate_access_token(identity=str(user['_id']))

    return jsonify({"message": "Login successful", "token": access_token}), 200

# ========================
# Login with Phone Number
# (OTP logic skipped for now)
# ========================
@bp.route('/login-phone', methods=['POST'])
def login_phone():
    data = request.get_json()
    phone = data.get('phone')

    user = UserModel.find_by_phone(phone)

    if not user:
        return jsonify({"message": "Phone number not registered"}), 401

    otp = "123456"  # Fixed OTP for now
    return jsonify({"message": "OTP sent", "otp": otp}), 200

@bp.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    phone = data.get('phone')
    otp = data.get('otp')

    if otp == "123456":
        user = UserModel.find_by_phone(phone)
        access_token = generate_access_token(identity=str(user['_id']))
        return jsonify({"message": "Login successful", "token": access_token}), 200
    else:
        return jsonify({"message": "Invalid OTP"}), 401

# ========================
# Profile - View
# ========================
@bp.route('/profile', methods=['GET'])
@jwt_required()
def profile():
    user_id = get_jwt_identity()
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})

    if not user:
        return jsonify({"message": "User not found"}), 404

    return jsonify({
        "name": user['name'],
        "email": user['email'],
        "phone": user['phone'],
        "created_at": user['created_at']
    }), 200

# ========================
# Profile - Update
# ========================
@bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    user_id = get_jwt_identity()
    data = request.get_json()

    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})

    if not user:
        return jsonify({"message": "User not found"}), 404

    updated_fields = {}
    if 'name' in data:
        updated_fields['name'] = data['name']
    if 'phone' in data:
        if not validate_phone(data['phone']):
            return jsonify({"message": "Invalid phone number. It must be exactly 10 digits"}), 400
        updated_fields['phone'] = data['phone']

    if updated_fields:
        mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": updated_fields})
        return jsonify({"message": "Profile updated successfully"}), 200
    else:
        return jsonify({"message": "No fields to update"}), 400

# ========================
# Change Password
# ========================
@bp.route('/change-password', methods=['PUT'])
@jwt_required()
def change_password():
    user_id = get_jwt_identity()
    data = request.get_json()

    old_password = data.get('old_password')
    new_password = data.get('new_password')

    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})

    if not user:
        return jsonify({"message": "User not found"}), 404

    if not UserModel.verify_password(user['password'], old_password):
        return jsonify({"message": "Old password is incorrect"}), 400

    if not validate_password(new_password):
        return jsonify({"message": "New password must contain letters and numbers and be at least 6 characters."}), 400

    hashed_password = generate_password_hash(new_password)

    mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"password": hashed_password}})

    return jsonify({"message": "Password changed successfully"}), 200

# ========================
# Forgot Password (NEW FEATURE)
# ========================
@bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data.get('email')

    user = UserModel.find_by_email(email)

    if not user:
        return jsonify({"message": "User with this email does not exist"}), 404

    return jsonify({"message": "Email found. Proceed to reset password."}), 200

# ========================
# Reset Password (NEW FEATURE)
# ========================
@bp.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    email = data.get('email')
    new_password = data.get('new_password')

    if not validate_password(new_password):
        return jsonify({"message": "New password must contain letters and numbers and be at least 6 characters."}), 400

    user = UserModel.find_by_email(email)

    if not user:
        return jsonify({"message": "User not found"}), 404

    hashed_password = generate_password_hash(new_password)

    mongo.db.users.update_one({"email": email}, {"$set": {"password": hashed_password}})
    return jsonify({"message": "Password reset successfully"}), 200

# ========================
# Test Database Connection
# ========================
@bp.route('/test_db')
def test_db():
    try:
        users = mongo.db.users.find_one()

        if users:
            return jsonify({"message": "Connected to MongoDB", "user": str(users)}), 200
        else:
            return jsonify({"message": "MongoDB connected, but no users found."}), 200
    except Exception as e:
        return jsonify({"message": f"Error connecting to MongoDB: {str(e)}"}), 500
