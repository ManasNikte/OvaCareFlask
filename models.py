from flask_login import UserMixin
from bson.objectid import ObjectId
from datetime import datetime

class User(UserMixin):
    def __init__(self, username, email, password, role, user_id, verification='pending', date_joined=None):
        self.username = username
        self.email = email
        self.password = password
        self.role = role
        self.id = user_id
        self.verification = verification  # Added for email verification status
        self.date_joined = date_joined or datetime.utcnow()  # Set to current time if not provided

    def get_id(self):
        return self.id
