from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from config import Config
from models import User
from bson.objectid import ObjectId, InvalidId
from datetime import datetime
from flask_mail import Mail, Message
import random
import string
import os
from werkzeug.utils import secure_filename
from pymongo import DESCENDING
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)
app.config.from_object(Config)



# Set the path to your model .pkl file
MODEL_PATH = './pcos1.pkl'  # Ensure this path is correct

model = None

# Load the model at the start of the Flask app
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
    else:
        print(f"Model not found at {MODEL_PATH}.")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# Feature name mapping based on your model's training
feature_name_mapping = {
    "age": "Age (yrs)",
    "weight": "Weight (Kg)",
    "height": "Height(Cm)",
    "bmi": "BMI",
    "blood_group": "Blood Group",
    "pulse_rate": "Pulse rate(bpm)",
    "rr": "RR (breaths/min)",
    "hb": "Hb(g/dl)",
    "cycle_type": "Cycle(R/I)",
    "cycle_length": "Cycle length(days)",
    "marriage_status": "Marraige Status (Yrs)",
    "pregnant": "Pregnant(Y/N)",
    "abortions": "No. of aborptions",
    "i_beta_hcg": "I   beta-HCG(mIU/mL)",
    "ii_beta_hcg": "II    beta-HCG(mIU/mL)",
    "fsh": "FSH(mIU/mL)",
    "lh": "LH(mIU/mL)",
    "fsh_lh_ratio": "FSH/LH",
    "hip": "Hip(inch)",
    "waist": "Waist(inch)",
    "waist_hip_ratio": "Waist:Hip Ratio",
    "tsh": "TSH (mIU/L)",
    "amh": "AMH(ng/mL)",
    "prl": "PRL(ng/mL)",
    "vit_d3": "Vit D3 (ng/mL)",
    "prg": "PRG(ng/mL)",
    "rbs": "RBS(mg/dl)",
    "weight_gain": "Weight gain(Y/N)",
    "hair_growth": "hair growth(Y/N)",
    "skin_darkening": "Skin darkening (Y/N)",
    "hair_loss": "Hair loss(Y/N)",
    "pimples": "Pimples(Y/N)",
    "fast_food": "Fast food (Y/N)",
    "reg_exercise": "Reg.Exercise(Y/N)",
    "bp_systolic": "BP _Systolic (mmHg)",
    "bp_diastolic": "BP _Diastolic (mmHg)",
    "follicle_no_l": "Follicle No. (L)",
    "follicle_no_r": "Follicle No. (R)",
    "avg_f_size_l": "Avg. F size (L) (mm)",
    "avg_f_size_r": "Avg. F size (R) (mm)",
    "endometrium": "Endometrium (mm)",
    "Sl. No_y": "Sl. No_y" 
}

# Blood group mapping (convert categorical to numeric)
blood_group_mapping = {
    "A": 0,
    "B": 1,
    "O": 2,
    "AB": 3
}

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict PCOS risk based on input data.
    """
    if model is None:
        return jsonify({"error": "Model not loaded", "message": "Prediction failed"}), 500

    # Define mappings for categorical values
    categorical_mappings = {
        "cycle_type": {"R": 1, "IR": 0},  # Regular (R) -> 1, Irregular (IR) -> 0
        "pregnant": {"Y": 1, "N": 0},     # Yes (Y) -> 1, No (N) -> 0
        "weight_gain": {"Y": 1, "N": 0},
        "hair_growth": {"Y": 1, "N": 0},
        "skin_darkening": {"Y": 1, "N": 0},
        "hair_loss": {"Y": 1, "N": 0},
        "pimples": {"Y": 1, "N": 0},
        "fast_food": {"Y": 1, "N": 0},
        "reg_exercise": {"Y": 1, "N": 0},
    }

    try:
        # Parse input data
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No data provided", "message": "Prediction failed"}), 400

        # Map input data to match model's features
        mapped_input_data = {}
        for key, value in input_data.items():
            if key in feature_name_mapping:
                if key == "blood_group":
                    # Convert blood group to numeric value
                    mapped_input_data[feature_name_mapping[key]] = blood_group_mapping.get(value, -1)
                elif key in categorical_mappings:
                    # Convert categorical feature to numeric code
                    mapped_input_data[feature_name_mapping[key]] = categorical_mappings[key].get(value, -1)
                else:
                    # Add all other features as they are
                    mapped_input_data[feature_name_mapping[key]] = value
            else:
                return jsonify({"error": f"Invalid feature: {key}", "message": "Prediction failed"}), 400

        # Convert mapped data into a DataFrame
        input_df = pd.DataFrame([mapped_input_data])

        # Perform prediction
        prediction = model.predict(input_df)

        # Prepare response
        response = {
            "prediction": int(prediction[0]),  # 0 (low risk) or 1 (high risk)
            "message": "Prediction successful"
        }
        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({"error": f"Value error: {ve}", "message": "Prediction failed"}), 500

    except Exception as e:
        return jsonify({"error": str(e), "message": "Prediction failed"}), 500




# Initialize extensions
mongo = PyMongo(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Define the folder to store uploaded blog images
UPLOAD_FOLDER = 'static/images/blogs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Configure mail settings
app.config['MAIL_SERVER'] = 'smtp.hostinger.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'manas.nikte@irrecordings.com'
app.config['MAIL_PASSWORD'] = 'Mcn78554714@'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_DEFAULT_SENDER'] = 'manas.nikte@irrecordings.com'

mail = Mail(app)

def send_email_notification(to_email, blog_title, comment):
    msg = Message(
        f'New Comment on Your Blog "{blog_title}"',
        recipients=[to_email]
    )
    msg.body = f"""
    Hello,

    A new comment has been added to your blog "{blog_title}":

    Comment by: {comment['author']}
    Comment: {comment['content']}
    Posted on: {comment['created_at'].strftime('%B %d, %Y at %I:%M %p')}

    Best Regards,
    Your Blog Team
    """
    mail.send(msg)


@login_manager.user_loader
def load_user(user_id):
    try:
        user_data = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        return User(user_data['username'], user_data['email'], user_data['password'], user_data['role'], str(user_data['_id']), user_data['verification'], user_data['date_joined']) if user_data else None
    except InvalidId:
        return None

# Error Handling
def access_denied():
    flash('Access denied.')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']

        if role not in ['user', 'doctor']:
            flash('Invalid role selected.')
            return redirect(url_for('register'))

        # Check if email is already registered
        if mongo.db.users.find_one({'email': email}):
            flash('Email already registered.')
            return redirect(url_for('register'))

        # Check if username is already taken
        if mongo.db.users.find_one({'username': username}):
            flash('Username already taken.')
            return redirect(url_for('register'))

        # Proceed with registration if email and username are unique
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        verification_code = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        user_id = mongo.db.users.insert_one({
            'username': username,
            'email': email,
            'password': hashed_password,
            'role': role,
            'verification': 'pending',
            'date_joined': datetime.utcnow()
        }).inserted_id

        verification_link = f"{Config.HOST_URL}/verify_email/{user_id}/{verification_code}"
        msg = Message("Email Verification", recipients=[email])
        msg.body = f"Please verify your email by clicking the following link: {verification_link}"
        mail.send(msg)

        flash('Registration successful! Please check your email for verification.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/has-pcos')
def has_pcos():
    return render_template('has-pcos.html')

@app.route('/no-pcos')
def no_pcos():
    return render_template('no-pcos.html')

@app.route('/verify_email/<user_id>/<code>')
def verify_email(user_id, code):
    # Implement verification logic here
    mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"verification": "completed"}})
    flash('Email verified successfully! You can now log in.')
    return redirect(url_for('login'))
    

@app.route('/resend_verification', methods=['GET', 'POST'])
def resend_verification():
    if request.method == 'POST':
        email = request.form['email']
        user_data = mongo.db.users.find_one({'email': email})

        if user_data and user_data['verification'] != 'completed':
            verification_code = ''.join(random.choices(string.ascii_letters + string.digits, k=16))  # Generate a new verification code
            mongo.db.users.update_one({"_id": user_data['_id']}, {"$set": {"verification": 'pending', 'verification_code': verification_code}})  # Reset verification status

            verify_link = f"{Config.HOST_URL}/verify_email/{user_data['_id']}/{verification_code}"  # Use the configured host URL
            msg = Message('Email Verification', recipients=[email])
            msg.body = f'Please verify your email by clicking the link: {verify_link}'
            mail.send(msg)
            flash('Verification email resent! Please check your inbox.')
        else:
            flash('Email not registered or already verified.')

    return render_template('resend_verification.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_data = mongo.db.users.find_one({'email': email})

        if user_data and bcrypt.check_password_hash(user_data['password'], password):
            user = User(user_data['username'], user_data['email'], user_data['password'], user_data['role'], str(user_data['_id']))
            login_user(user)
            flash('Login successful!')

            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user.role == 'user':
                return redirect(url_for('user_dashboard'))
            elif user.role == 'doctor':
                return redirect(url_for('doctor_dashboard'))
        else:
            flash('Invalid email or password')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    print("Logout function called.")  # Debugging statement
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/')
def home():
    # Fetch the latest 3 public blogs from the database
    recent_blogs = mongo.db.posts.find({"visibility": "public"}).sort("created_at", -1).limit(3)
    return render_template('home.html', recent_blogs=recent_blogs)

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        return access_denied()
    return render_template('admin_dashboard.html', username=current_user.username)

@app.route('/user/dashboard')
@login_required
def user_dashboard():
    if current_user.role != 'user':
        return access_denied()
    return render_template('user_dashboard.html', username=current_user.username)

@app.route('/doctor/dashboard')
@login_required
def doctor_dashboard():
    if current_user.role != 'doctor':
        return access_denied()
    return render_template('doctor_dashboard.html', username=current_user.username)

@app.route('/blogs')
def blogs():
    page = request.args.get('page', 1, type=int)  # Get the current page number from query parameters
    per_page = 4  # Number of blogs per page
    offset = (page - 1) * per_page  # Calculate offset for pagination

    # Fetch the total number of blogs for pagination
    total_blogs = mongo.db.posts.count_documents({'visibility': 'public'})
    total_pages = (total_blogs + per_page - 1) // per_page  # Calculate total pages

    # Fetch the latest blogs first with pagination
    blogs = mongo.db.posts.find({'visibility': 'public'}).sort('created_at', -1).skip(offset).limit(per_page)

    return render_template('blogs.html', blogs=blogs, page=page, total_blogs=total_blogs, per_page=per_page, total_pages=total_pages)

@app.route('/blog/<blog_id>', methods=['GET', 'POST'])
def blog_details(blog_id):
    blog = mongo.db.posts.find_one({'_id': ObjectId(blog_id)})

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        content = request.form['content']
        
        # Create a new comment with a timestamp
        comment = {
            '_id': ObjectId(), 
            'author': name,
            'email': email,
            'content': content,
            'created_at': datetime.utcnow()
        }
        
        # Push the comment to the blog's comments array
        mongo.db.posts.update_one(
            {'_id': ObjectId(blog_id)},
            {'$push': {'comments': comment}}
        )
        
        # Notify the blog author via email
        author_username = blog['author']  # Assuming 'author' field holds the username
        author = mongo.db.users.find_one({'username': author_username})
        if author:
            author_email = author.get('email')
            if author_email:
                send_email_notification(author_email, blog['title'], comment)

        return redirect(url_for('blog_details', blog_id=blog_id))

    # Fetch the latest 3 comments initially
    comments = blog['comments'][-3:][::-1]

    return render_template('blog_details.html', blog=blog, comments=comments)

@app.route('/load_more_comments/<blog_id>', methods=['GET'])
def load_more_comments(blog_id):
    skip = int(request.args.get('skip', 3))  # Initially skip 3 comments
    limit = int(request.args.get('limit', 5))  # Load 5 more comments by default
    blog = mongo.db.posts.find_one({'_id': ObjectId(blog_id)})

    total_comments = len(blog['comments'])
    # Get the next batch of comments, skipping the already loaded ones
    more_comments = blog['comments'][-(skip + limit):-skip][::-1] if skip < total_comments else []

    # Convert ObjectId to string for JSON serialization
    more_comments_serialized = []
    for comment in more_comments:
        comment['_id'] = str(comment['_id'])  # Convert ObjectId to string
        more_comments_serialized.append(comment)

    # Determine if there are more comments to load
    more_available = (skip + limit) < total_comments

    return jsonify({'comments': more_comments_serialized, 'more_available': more_available})

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/detection_advanced')
def detection_advanced():
    return render_template('detection_advanced.html')

# Dummy data for Yoga and Nutrition
yoga_data = [
    {"title": f"Yoga {i}", "description": f"Description for Yoga {i}", "image": "https://via.placeholder.com/400x250", "link": "#"}
    for i in range(1, 10)
]

nutrition_data = [
    {"title": f"Nutrition {i}", "description": f"Description for Nutrition {i}", "image": "https://via.placeholder.com/400x250", "link": "#"}
    for i in range(1, 10)
]

# Pagination logic
def paginate(items, page, per_page=9):
    total_pages = (len(items) + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end], total_pages

@app.route('/yoga', defaults={'page': 1})
@app.route('/yoga/page/<int:page>')
def yoga(page=1):
    # Yoga items
    yoga_items = [
        {"title": "15-Minute Morning Yoga for PCOS", "description": "A quick and effective morning yoga routine to help manage PCOS symptoms.", "image": "https://img.youtube.com/vi/H4B3xSCpEfo/0.jpg", "link": "https://www.youtube.com/watch?v=H4B3xSCpEfo"},
        {"title": "Relaxing Yoga for PCOS and Hormonal Imbalances", "description": "A calming yin yoga session to balance hormones and improve well-being.", "image": "https://img.youtube.com/vi/arnokQ2ZbO8/0.jpg", "link": "https://www.youtube.com/watch?v=arnokQ2ZbO8"},
        {"title": "30-Minute Yoga for PCOS", "description": "A comprehensive 30-minute session tailored for PCOS management.", "image": "https://img.youtube.com/vi/WEJX48O1izk/0.jpg", "link": "https://www.youtube.com/watch?v=WEJX48O1izk"},
        {"title": "Yoga for PCOS: A Mindful Practice", "description": "A 45-minute yoga session focusing on mindfulness and PCOS relief.", "image": "https://img.youtube.com/vi/H31HgCvovu0/0.jpg", "link": "https://www.youtube.com/watch?v=H31HgCvovu0"},
        {"title": "30-Minute Yoga for Hormonal Balance", "description": "Effective asanas for irregular periods and hormonal imbalances.", "image": "https://img.youtube.com/vi/XsvsWdA6xqc/0.jpg", "link": "https://www.youtube.com/watch?v=XsvsWdA6xqc"},
        {"title": "Yoga for PCOS: Manage Irregular Periods", "description": "Yoga poses to help regulate periods and balance hormones.", "image": "https://img.youtube.com/vi/L7MyZKX8-mY/0.jpg", "link": "https://www.youtube.com/watch?v=L7MyZKX8-mY"},
        {"title": "25-Minute Yoga for PCOS by Shilpa Shetty", "description": "A session by Shilpa Shetty tailored for PCOS wellness.", "image": "https://img.youtube.com/vi/AkN_nCpJ2C0/0.jpg", "link": "https://www.youtube.com/watch?v=AkN_nCpJ2C0"},
        {"title": "PCOS Yoga for Hormone Balance", "description": "A yoga routine to harmonize hormonal health.", "image": "https://img.youtube.com/vi/DrVVfGXaM5k/0.jpg", "link": "https://www.youtube.com/watch?v=DrVVfGXaM5k"},
        {"title": "Yoga + Pilates for PCOS", "description": "A fusion of yoga and pilates for PCOS-friendly weight loss.", "image": "https://img.youtube.com/vi/c57ksNThbKQ/0.jpg", "link": "https://www.youtube.com/watch?v=c57ksNThbKQ"},
        {"title": "Yoga for Women's Health", "description": "A specialized session for PCOS, endometriosis, and infertility.", "image": "https://img.youtube.com/vi/VjRXtDhBUpA/0.jpg", "link": "https://www.youtube.com/watch?v=VjRXtDhBUpA"},
        {"title": "PCOS Yoga by Yogalates with Rashmi", "description": "A practical approach to PCOS management through yogalates.", "image": "https://img.youtube.com/vi/hNkqaMw8YCc/0.jpg", "link": "https://www.youtube.com/watch?v=hNkqaMw8YCc"},
        {"title": "Yoga for PCOS: Day 1", "description": "The first day of a six-day course for managing PCOS.", "image": "https://img.youtube.com/vi/Gj_d9ueu6Y4/0.jpg", "link": "https://www.youtube.com/watch?v=Gj_d9ueu6Y4"},   
    ]

    # Pagination logic
    per_page = 9
    total_items = len(yoga_items)
    total_pages = (total_items + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    yoga_items_page = yoga_items[start:end]
    
    return render_template("yoga.html", yoga_items=yoga_items_page, page=page, total_pages=total_pages)


@app.route('/nutrition', defaults={'page': 1})
@app.route('/nutrition/page/<int:page>')
def nutrition(page=1):
    # Nutrition items for PCOS
    nutrition_items = [
        {
            "title": "PCOS Diet Plan to Lose Weight",
            "description": "This video explores a practical day-in-the-life meal plan for weight loss and PCOS management.",
            "image": "https://img.youtube.com/vi/vSAp1oafrxg/0.jpg",
            "link": "https://www.youtube.com/watch?v=vSAp1oafrxg"
        },
        {
            "title": "Best Diet for PCOS | Educational Video",
            "description": "A detailed guide to the best dietary practices for those with PCOS, presented by Biolayne.",
            "image": "https://img.youtube.com/vi/qfGekqFJqPc/0.jpg",
            "link": "https://www.youtube.com/watch?v=qfGekqFJqPc"
        },
        {
            "title": "What I Eat in a Day for PCOS",
            "description": "A realistic meal plan video showing food choices for PCOS-friendly weight management.",
            "image": "https://img.youtube.com/vi/f6K2a-NM3Xk/0.jpg",
            "link": "https://www.youtube.com/watch?v=f6K2a-NM3Xk"
        },
        {
            "title": "PCOS Weight Loss Meal Plan Ideas",
            "description": "Offers meal ideas to help control PCOS symptoms and lose weight effectively.",
            "image": "https://img.youtube.com/vi/SUFqUCZGjrI/0.jpg",
            "link": "https://www.youtube.com/watch?v=SUFqUCZGjrI"
        },
        {
            "title": "PCOS-Friendly Smoothie Recipes",
            "description": "Tips and recipes for creating PCOS-friendly smoothies rich in nutrients.",
            "image": "https://img.youtube.com/vi/kHRlNRLMNIA/0.jpg",
            "link": "https://www.youtube.com/watch?v=kHRlNRLMNIA"
        },
        {
            "title": "Top Foods to Avoid with PCOS",
            "description": "An informative guide to foods that may worsen PCOS symptoms.",
            "image": "https://img.youtube.com/vi/Q4IjcUD9xlU/0.jpg",
            "link": "https://www.youtube.com/watch?v=Q4IjcUD9xlU"
        },
        {
            "title": "How to Balance Hormones with Food",
            "description": "Explores how to naturally balance hormones through diet, focusing on PCOS.",
            "image": "https://img.youtube.com/vi/OrJ6Jl69YXY/0.jpg",
            "link": "https://www.youtube.com/watch?v=OrJ6Jl69YXY"
        },
        {
            "title": "Nutrition Tips for PCOS Weight Loss",
            "description": "Key dietary tips for managing PCOS and achieving sustainable weight loss.",
            "image": "https://img.youtube.com/vi/k9C6EVKu0tQ/0.jpg",
            "link": "https://www.youtube.com/watch?v=k9C6EVKu0tQ"
        },
        {
            "title": "High-Protein PCOS Meal Plan",
            "description": "Discusses high-protein meal options to improve insulin sensitivity and support weight loss.",
            "image": "https://img.youtube.com/vi/YrDwv2ZHgFg/0.jpg",
            "link": "https://www.youtube.com/watch?v=YrDwv2ZHgFg"
        },
        {
            "title": "Indian Diet Plan for PCOS",
            "description": "A specific focus on Indian dietary options that can help manage PCOS.",
            "image": "https://img.youtube.com/vi/8v7OJ4CHboc/0.jpg",
            "link": "https://www.youtube.com/watch?v=8v7OJ4CHboc"
        }
    ]

    # Pagination logic
    per_page = 9
    total_items = len(nutrition_items)
    total_pages = (total_items + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    nutrition_items_page = nutrition_items[start:end]
    
    return render_template("nutrition.html", nutrition_items=nutrition_items_page, page=page, total_pages=total_pages)

@app.route('/user/blogs')
@login_required
def user_blogs():
    if current_user.role != 'user':
        return access_denied()
    
    # Pagination parameters
    page = int(request.args.get('page', 1))  # Default to page 1
    per_page = 8  # Number of blogs per page
    
    # Search parameters
    search_query = request.args.get('search', '')  # Default to empty string
    search_field = request.args.get('search_field', 'title')  # Default search by title
    
    query = {"author": current_user.username}  # Search only for the current user's blogs

    # Search logic
    if search_query:
        if search_field == 'title':
            query['title'] = {'$regex': search_query, '$options': 'i'}  # Case-insensitive search by title
        elif search_field == 'content':
            query['content'] = {'$regex': search_query, '$options': 'i'}  # Search in content
        elif search_field == 'created_at':
            try:
                # Parse date and search for exact matches
                search_date = datetime.strptime(search_query, '%Y-%m-%d')
                query['created_at'] = {'$gte': search_date, '$lt': search_date.replace(hour=23, minute=59, second=59)}
            except ValueError:
                pass  # Ignore invalid date formats

    # Get total number of blogs for pagination
    total_blogs = mongo.db.posts.count_documents(query)
    
    # Fetch blogs for the current page
    blogs = (mongo.db.posts.find(query)
             .sort('created_at', -1)  # Sort by newest
             .skip((page - 1) * per_page)
             .limit(per_page))
    
    total_pages = (total_blogs + per_page - 1) // per_page  # Calculate total pages
    
    # Calculate the range of pages to display
    page_range = 2  # Number of pages to show before and after the current page
    start_page = max(1, page - page_range)
    end_page = min(total_pages, page + page_range)

    return render_template('user_blogs.html', blogs=blogs, total_pages=total_pages, current_page=page, 
                           search_query=search_query, search_field=search_field, 
                           start_page=start_page, end_page=end_page)

@app.route('/admin/blogs')
@login_required
def admin_blogs():
    if current_user.role != 'admin':
        return access_denied()
    
    # Pagination parameters
    page = int(request.args.get('page', 1))  # Default to page 1
    per_page = 8  # Number of blogs per page
    
    # Search parameters
    search_query = request.args.get('search', '')  # Default to empty string
    search_field = request.args.get('search_field', 'author')  # Default search by author
    
    query = {}
    
    # Search logic (same as before)
    if search_query:
        if search_field == 'author':
            query['author'] = {'$regex': search_query, '$options': 'i'}  # Case-insensitive search
        elif search_field == 'content':
            query['content'] = {'$regex': search_query, '$options': 'i'}  # Search in content
        elif search_field == 'created_at':
            try:
                # Parse date and search for exact matches
                search_date = datetime.strptime(search_query, '%Y-%m-%d')
                query['created_at'] = {'$gte': search_date, '$lt': search_date.replace(hour=23, minute=59, second=59)}
            except ValueError:
                pass  # Ignore invalid date formats
    
    # Get total number of blogs for pagination
    total_blogs = mongo.db.posts.count_documents(query)
    
    # Fetch blogs for the current page
    blogs = (mongo.db.posts.find(query)
             .sort('created_at', -1)  # Sort by newest
             .skip((page - 1) * per_page)
             .limit(per_page))
    
    total_pages = (total_blogs + per_page - 1) // per_page  # Calculate total pages
    
    # Calculate the range of pages to display
    page_range = 2  # Number of pages to show before and after the current page
    start_page = max(1, page - page_range)
    end_page = min(total_pages, page + page_range)

    return render_template('admin_blogs.html', blogs=blogs, total_pages=total_pages, current_page=page, 
                           search_query=search_query, search_field=search_field, 
                           start_page=start_page, end_page=end_page)

@app.route('/preview_blog/<post_id>', methods=['GET', 'POST'])
@login_required
def preview_blog(post_id):
    blog = mongo.db.posts.find_one({'_id': ObjectId(post_id)})

    if not blog:
        flash('Blog not found!', 'danger')
        return redirect(url_for('admin_blogs'))  # Redirect to admin blogs if not found

    # Check user role and ownership
    if current_user.role == 'user' and blog['author'] != current_user.username:
        flash('Access denied: You do not have permission to preview this blog.', 'danger')
        return redirect(url_for('user_blogs'))  # Redirect to user's blogs

    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        image = request.files['image']

        # Handle image upload
        if image and allowed_file(image.filename):
            # Save the image securely
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            filename = blog['image']  # Retain the old image if no new image is uploaded

        # Update the blog post in the database
        mongo.db.posts.update_one(
            {'_id': ObjectId(post_id)},
            {
                '$set': {
                    'title': title,
                    'content': content,
                    'image': filename,  # Update the image filename
                    'visibility': 'private',
                    'updated_at': datetime.utcnow()  # Optional: track when it was updated
                }
            }
        )
        flash('Blog updated successfully!', 'success')

        # Redirect based on user role
        if current_user.role == 'admin':
            return redirect(url_for('admin_blogs'))
        else:
            return redirect(url_for('user_blogs'))

    return render_template('preview_blog.html', blog=blog)

@app.route('/delete_comment/<comment_id>', methods=['POST'])
@login_required
def delete_comment(comment_id):
    # Check if the user has permission to delete the comment
    blog = mongo.db.posts.find_one({'comments._id': ObjectId(comment_id)})

    if not blog:
        flash('Comment not found!', 'danger')
        return redirect(url_for('preview_blog', post_id=blog['_id']))  # Redirect to the preview_blog

    # Perform the deletion
    mongo.db.posts.update_one(
        {'_id': blog['_id']},
        {'$pull': {'comments': {'_id': ObjectId(comment_id)}}}
    )

    flash('Comment deleted successfully!', 'success')
    return redirect(url_for('preview_blog', post_id=blog['_id']))  # Redirect to the preview_blog after deletion


@app.route('/add_blog', methods=['GET', 'POST'])
@login_required
def add_blog():
    if current_user.role != 'user':
        return access_denied()
    
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        image = request.files['image']

        if image and allowed_file(image.filename):
            # Save the image securely
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Insert blog post into database with image and created_at timestamp
            mongo.db.posts.insert_one({
                'title': title,
                'content': content,
                'author': current_user.username,
                'image': filename,  # Save the image filename
                'visibility': 'private',
                'approved': False,
                'created_at': datetime.utcnow(),  # Store date and time of creation
                'comments': []  # Initialize an empty array for comments
            })

            flash('Post submitted for approval!')
            return redirect(url_for('user_blogs'))
        else:
            flash('Invalid image format! Please upload PNG, JPG, JPEG, or GIF.')

    return render_template('add_blog.html')

@app.route('/approve_blog/<post_id>', methods=['POST'])
@login_required
def approve_blog(post_id):
    if current_user.role != 'admin':
        return access_denied()

    mongo.db.posts.update_one({"_id": ObjectId(post_id)}, {"$set": {"visibility": "public", "approved": True}})
    flash('Blog post approved and made public!')
    return redirect(url_for('admin_blogs'))

@app.route('/toggle_visibility/<blog_id>')
@login_required
def toggle_visibility(blog_id):
    if current_user.role != 'admin':
        return access_denied()

    blog = mongo.db.posts.find_one({"_id": ObjectId(blog_id)})
    if blog:
        new_visibility = 'public' if blog['visibility'] == 'private' else 'private'
        mongo.db.posts.update_one({"_id": ObjectId(blog_id)}, {"$set": {"visibility": new_visibility}})
        flash('Blog visibility updated.')
    return redirect(url_for('admin_blogs'))

@app.route('/delete_blog/<blog_id>')
@login_required
def delete_blog(blog_id):
    if current_user.role != 'admin':
        return access_denied()

    mongo.db.posts.delete_one({"_id": ObjectId(blog_id)})
    flash('Blog deleted successfully.')
    return redirect(url_for('admin_blogs'))

@app.route('/admin/profile', methods=['GET', 'POST'])
@login_required
def admin_profile():
    if current_user.role != 'admin':
        return access_denied()

    admin = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})

    if request.method == 'POST':
        email = request.form['email']

        existing_user = mongo.db.users.find_one({"email": email})
        if existing_user and existing_user['_id'] != ObjectId(current_user.id):
            flash('Email already exists. Please use a different email.')
        else:
            # Update email and set verification to pending
            mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": {"email": email, "verification": "pending"}})
            verification_code = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": {"verification_code": verification_code}})
            verify_link = f"{Config.HOST_URL}/verify_email/{current_user.id}/{verification_code}"

            # Send verification email
            msg = Message('Email Verification', recipients=[email])
            msg.body = f'Please verify your email by clicking the link: {verify_link}'
            mail.send(msg)

            flash('Profile updated successfully! Please verify your new email.')
            logout_user()  # Log out the user after email update
            return redirect(url_for('login'))  # Redirect to login page

    return render_template('admin_profile.html', user=admin, username=current_user.username)

@app.route('/user/profile', methods=['GET', 'POST'])
@login_required
def user_profile():
    if current_user.role != 'user':
        return access_denied()

    user = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})

    if request.method == 'POST':
        email = request.form['email']

        existing_user = mongo.db.users.find_one({"email": email})
        if existing_user and existing_user['_id'] != ObjectId(current_user.id):
            flash('Email already exists. Please use a different email.')
        else:
            # Update email and set verification to pending
            mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": {"email": email, "verification": "pending"}})
            verification_code = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": {"verification_code": verification_code}})
            verify_link = f"{Config.HOST_URL}/verify_email/{current_user.id}/{verification_code}"

            # Send verification email
            msg = Message('Email Verification', recipients=[email])
            msg.body = f'Please verify your email by clicking the link: {verify_link}'
            mail.send(msg)

            flash('Profile updated successfully! Please verify your new email.')
            logout_user()  # Log out the user after email update
            return redirect(url_for('login'))  # Redirect to login page

    return render_template('user_profile.html', user=user, username=current_user.username)

@app.route('/doctor/profile', methods=['GET', 'POST'])
@login_required
def doctor_profile():
    if current_user.role != 'doctor':
        return access_denied()

    user = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})

    if request.method == 'POST':
        email = request.form['email']

        existing_user = mongo.db.users.find_one({"email": email})
        if existing_user and existing_user['_id'] != ObjectId(current_user.id):
            flash('Email already exists. Please use a different email.')
        else:
            # Update email and set verification to pending
            mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": {"email": email, "verification": "pending"}})
            verification_code = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": {"verification_code": verification_code}})
            verify_link = f"{Config.HOST_URL}/verify_email/{current_user.id}/{verification_code}"

            # Send verification email
            msg = Message('Email Verification', recipients=[email])
            msg.body = f'Please verify your email by clicking the link: {verify_link}'
            mail.send(msg)

            flash('Profile updated successfully! Please verify your new email.')
            logout_user()  # Log out the user after email update
            return redirect(url_for('login'))  # Redirect to login page

    return render_template('doctor_profile.html', user=user, username=current_user.username)

@app.route('/doctor/appointments')
@login_required
def doctor_appointments():
    if current_user.role != 'doctor':
        return access_denied()
    return render_template('doctor_appointments.html')

@app.route('/doctor/ratings')
@login_required
def doctor_ratings():
    if current_user.role != 'doctor':
        return access_denied()
    return render_template('doctor_ratings.html')

@app.route('/update_password', methods=['POST'])
@login_required
def update_password():
    if current_user.role not in ['user', 'admin', 'doctor']:
        return access_denied()

    current_password = request.form['current_password']
    new_password = request.form['new_password']
    re_enter_new_password = request.form['re-enter_new_password']

    user_data = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})

    if user_data and bcrypt.check_password_hash(user_data['password'], current_password):
        if new_password == re_enter_new_password:
            hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
            mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": {"password": hashed_password}})
            flash('Password updated successfully!')
        else:
            flash('New passwords do not match. Please try again.')
    else:
        flash('Current password is incorrect.')

    if current_user.role == 'user':
        return redirect(url_for('user_profile'))
    elif current_user.role == 'admin':
        return redirect(url_for('admin_profile'))

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = mongo.db.users.find_one({"email": email})

        if user:
            otp = generate_otp()
            mongo.db.users.update_one({"_id": ObjectId(user['_id'])}, {"$set": {"otp": otp}})

            msg = Message("Password Reset OTP", recipients=[email])
            msg.body = f"Your OTP for password reset is {otp}."
            mail.send(msg)

            flash('OTP sent to your email.')
            return redirect(url_for('reset_password', email=email))
        else:
            flash('Email not registered.')

    return render_template('forgot_password.html')

@app.route('/reset_password/<email>', methods=['GET', 'POST'])
def reset_password(email):
    if request.method == 'POST':
        otp = request.form['otp']
        new_password = request.form['new_password']
        re_enter_new_password = request.form['re-enter_new_password']

        user = mongo.db.users.find_one({"email": email})

        if user and user['otp'] == otp:
            if new_password == re_enter_new_password:
                hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
                mongo.db.users.update_one({"_id": ObjectId(user['_id'])}, {"$set": {"password": hashed_password, "otp": None}})
                flash('Password reset successful!')
                return redirect(url_for('login'))
            else:
                flash('New passwords do not match.')
        else:
            flash('Invalid OTP.')

    return render_template('forgot_password.html', email=email)

@app.route('/users_list', methods=['GET'])
def users_list():
    if current_user.role != 'admin':
        return access_denied()
    
    page = int(request.args.get('page', 1))
    per_page = 10
    query = request.args.get('query', '').strip()

    # Create the query filter
    filter = {"role": "user"}
    if query:
        filter["$or"] = [
            {"username": {"$regex": query, "$options": "i"}},
            {"email": {"$regex": query, "$options": "i"}}
        ]

    # Count the total number of matching documents
    total = mongo.db.users.count_documents(filter)

    # Get the users with pagination and sorting
    users_cursor = mongo.db.users.find(filter).sort("date_joined", -1).skip((page - 1) * per_page).limit(per_page)

    users = list(users_cursor)
    
    return render_template('user_list.html', users=users, page=page, per_page=per_page, total=total)

@app.route('/doctors_list', methods=['GET'])
def doctors_list():
    if current_user.role != 'admin':
        return access_denied()
    
    page = int(request.args.get('page', 1))
    per_page = 10
    query = request.args.get('query', '').strip()

    # Create the query filter
    filter = {"role": "doctor"}
    if query:
        filter["$or"] = [
            {"username": {"$regex": query, "$options": "i"}},
            {"email": {"$regex": query, "$options": "i"}}
        ]

    # Count the total number of matching documents
    total = mongo.db.users.count_documents(filter)

    # Get the doctors with pagination and sorting
    doctors_cursor = mongo.db.users.find(filter).sort("date_joined", -1).skip((page - 1) * per_page).limit(per_page)

    doctors = list(doctors_cursor)

    return render_template('doctors_list.html', doctors=doctors, page=page, per_page=per_page, total=total)


# if __name__ == '__main__':
#     app.run(debug=True)
