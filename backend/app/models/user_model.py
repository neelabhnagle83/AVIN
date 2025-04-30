from werkzeug.security import generate_password_hash, check_password_hash
from app import mongo
from bson import ObjectId
from datetime import datetime

class UserModel:
    @staticmethod
    def create_user(email, name, phone, password, onboarding_info=None):
        hashed_password = generate_password_hash(password)
        user = {
            "email": email,
            "name": name,
            "phone": phone,
            "password": hashed_password,
            "created_at": datetime.utcnow(),
            "onboarding_info": onboarding_info or {}  # Default to empty if no info
        }
        mongo.db.users.insert_one(user)
        return user

    @staticmethod
    def update_onboarding_info(user_id, onboarding_info):
        result = mongo.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"onboarding_info": onboarding_info}}
        )
        return result.modified_count > 0

    @staticmethod
    def find_by_email(email):
        return mongo.db.users.find_one({"email": email})

    @staticmethod
    def find_by_phone(phone):
        return mongo.db.users.find_one({"phone": phone})

    @staticmethod
    def verify_password(stored_password, provided_password):
        return check_password_hash(stored_password, provided_password)
