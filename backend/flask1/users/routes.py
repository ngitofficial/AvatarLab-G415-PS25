import os
from flask import Blueprint, request, jsonify, Blueprint
from flask_login import login_user, current_user, logout_user, login_required
from flask1 import db, bcrypt
from flask1.models import User

users = Blueprint('users', __name__)

@users.route("/register", methods=['POST'])
def register():
    if current_user.is_authenticated:
        return jsonify({ "message": "Already logged in" }), 200

    data = request.get_json()

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username is already taken'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already exists'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    new_user = User(username=username, email=email, password=hashed_password)

    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Registration failed', 'details': str(e)}), 500


@users.route("/login", methods=['POST'])
def login():
    if current_user.is_authenticated:
        return jsonify({"message": "Already logged in"}), 200

    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    remember = data.get("remember", False)

    if not email or not password:
        return jsonify({"message": "Email and password are required"}), 400

    user = User.query.filter_by(email=email).first()

    if user and bcrypt.check_password_hash(user.password, password):
        login_user(user, remember=remember)
        return jsonify({
            "message": "Login successful",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email
            }
        }), 200
    else:
        return jsonify({
            "message": "Invalid email or password"
        }), 401
    
@users.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Successfully logged out"}), 200

@users.route("/check_auth")
@login_required
def check_auth():
    return jsonify({"authenticated": current_user.is_authenticated})
