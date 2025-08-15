from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_file, Response,flash
from flask_sqlalchemy import SQLAlchemy
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import os
import uuid
from datetime import datetime
import random
from werkzeug.security import generate_password_hash, check_password_hash
import string
import base64

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

# Configure the SQLAlchemy part
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Define Upload Folder
UPLOAD_FOLDER = 'static/input_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define a model for storing detected classes
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), nullable=False)
    media_type = db.Column(db.Enum('image', 'video', name='media_type_enum'),nullable=False)  # 'image' or 'video'
    input_file_path = db.Column(db.String(300), nullable=False)  # Store uploaded file path
    confidence = db.Column(db.Float, nullable=False)
    login_time = db.Column(db.DateTime, nullable=False)
    logout_time = db.Column(db.DateTime, nullable=True)
    detected_classes = db.Column(db.Text, nullable=True)
    report = db.Column(db.String(200), nullable=True)  # Link to recent activity
    satisfactory = db.Column(db.Enum('Yes', 'No',name='satisfactory_enum'), nullable=True)  # User feedback
    
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    otp = db.Column(db.String(6), nullable=True)  # New column
    
class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Create the database tables
with app.app_context():
    db.create_all()

# Load the YOLOv8 model (pretrained or custom)
model = YOLO("yolov8n.pt")  # You can replace this with the path to your custom model

# Function to preprocess image
def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode == 'L':
        img = img.convert('RGB')
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# Logged in Checker
def login_required(f):
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

# Enhanced `is_box_inside` function
def is_box_inside(box1, box2):
    """
    Check if box1 is entirely inside box2 with stricter checks based on area ratios.
    """
    x1, y1, x2, y2 = box1
    px1, py1, px2, py2 = box2

    # Check if all corners of box1 are within box2
    is_fully_inside = (
        x1 >= px1 and y1 >= py1 and x2 <= px2 and y2 <= py2
    )

    # Compute areas for additional validation
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (px2 - px1) * (py2 - py1)

    # Validate containment with area ratio (box1 must be significantly smaller)
    area_ratio_threshold = 0.8  # At least 20% smaller to be considered "inside"
    is_significantly_smaller = (area_box1 / area_box2) < area_ratio_threshold

    # Only classify as inside if both conditions are met
    return is_fully_inside and is_significantly_smaller


# Function to draw detections with parent-child relationships
def draw_detections_with_index(image, predictions):
    """
    Draw bounding boxes with parent-child relationships based on stricter containment checks.
    Ensures unique labels and resolves conflicts for overlapping bounding boxes by keeping the highest confidence label.
    """
    class_counter = {}  # To track the index for each class
    detected_classes = []
    processed_boxes = {}  # To store unique boxes for deduplication

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Step 1: Filter predictions to ensure unique bounding boxes
    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred[:4])  # Bounding box coordinates
        confidence = pred[4]  # Confidence score
        class_id = int(pred[5])  # Class ID
        label = model.names[class_id]  # Class name

        # Create a unique identifier for the box based on coordinates
        box_id = (x1, y1, x2, y2)

        # If the box already exists, keep the label with the highest confidence
        if box_id in processed_boxes:
            if processed_boxes[box_id]["confidence"] < confidence:
                processed_boxes[box_id] = {"label": label, "confidence": confidence}
        else:
            processed_boxes[box_id] = {"label": label, "confidence": confidence}

    # Step 2: Assign unique indices and handle nesting relationships
    bounding_boxes = []
    for box_id, details in processed_boxes.items():
        x1, y1, x2, y2 = box_id
        label = details["label"]

        # Ensure the class_counter entry exists for the label
        class_counter.setdefault(label, 0)
        class_counter[label] += 1

        # Assign a unique label with an index
        label_with_index = f"{label} #{class_counter[label]}"
        bounding_boxes.append([label, label_with_index, [x1, y1, x2, y2]])

    # Step 3: Handle parent-child relationships and draw bounding boxes
    for i, (label, label_with_index, box) in enumerate(bounding_boxes):
        parent_label_with_index = None

        # Check if this box is inside another larger box
        for j, (parent_label, parent_label_index, parent_box) in enumerate(bounding_boxes):
            if i != j and is_box_inside(box, parent_box):
                parent_label_with_index = parent_label_index
                break

        # Update the label if a parent box is found
        if parent_label_with_index:
            label_with_index = f"{label} for {parent_label_with_index}"

        # Draw the bounding box
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate text size for background
        font_scale = 0.7
        text_thickness = 2
        text_size = cv2.getTextSize(label_with_index, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        text_width, text_height = text_size

        # Adjust text background coordinates with padding
        text_background_x1 = x1
        text_background_y1 = y1 - text_height - 15
        text_background_x2 = x1 + text_width + 10
        text_background_y2 = y1

        # Ensure text stays within image width
        if text_background_x2 > image_width:  # If text goes out of the right boundary
            text_background_x1 = max(0, image_width - text_width - 15)  # Shift text to the left
            text_background_x2 = image_width - 5

        if text_background_y1 < 0:  # If text goes out of the top boundary
            text_background_y1 = y1 + 5  # Shift below the box
            text_background_y2 = y1 + text_height + 15

        # Draw a filled rectangle for text background
        cv2.rectangle(
            image,
            (text_background_x1, text_background_y1),
            (text_background_x2, text_background_y2),
            (0, 255, 0),
            thickness=cv2.FILLED,
        )

        # Add the label text on top of the background
        cv2.putText(
            image,
            label_with_index,
            (text_background_x1 + 5, text_background_y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text for contrast
            thickness=text_thickness,
        )

        detected_classes.append(label_with_index)

    return image, detected_classes

# Home route with upload form
@app.route('/')
def home():
    return render_template('index.html', random_num=random.random())

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/forgot_pass')
def forgot_pass():
    return render_template('forgot_pass.html')

@app.route('/user_list')
def user_list():
    # Fetch all users except the admin
    users = User.query.filter(User.email != "admin@gmail.com").all()
    
    # Fetch the latest login and logout times for each user
    for user in users:
        latest_detection = Detection.query.filter_by(username=user.email).order_by(Detection.login_time.desc()).first()
        if latest_detection:
            user.last_login = latest_detection.login_time
            user.last_logout = latest_detection.logout_time
        else:
            user.last_login = None
            user.last_logout = None

    return render_template('user_list.html', users=users)

@app.route('/detections_list')
def detections_list():
    # Fetch all detections with user relationship
    detections = db.session.query(
        Detection,
        User.full_name,
        User.email
    ).join(User, Detection.username == User.email)\
     .order_by(Detection.login_time.desc())\
     .all()

    return render_template('detections_list.html', detections=detections)

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']
        
        # Hash the password before storing it
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        new_user = User(full_name=full_name, email=email, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('User added successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding user: {str(e)}', 'danger')
        
        return redirect(url_for('user_list'))
    return render_template('add_user.html')


@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    
    try:
        db.session.delete(user)
        db.session.commit()
        flash('User deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting user: {str(e)}', 'danger')
    
    return redirect(url_for('user_list'))

@app.route('/contact_list')
def contact_list():
    # Fetch all contact messages
    contact_messages = ContactMessage.query.order_by(ContactMessage.timestamp.desc()).all()
    return render_template('contact_list.html', contact_messages=contact_messages)

# Sign Up route
@app.route('/signup', methods=['POST'])
def signup():
    full_name = request.form['fullName']
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirmPassword']

    # Check if the email already exists
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return render_template('login.html', signup_error="Email already exists")

    # Check if passwords match
    if password != confirm_password:
        return render_template('login.html', signup_error="Passwords do not match")

    # Hash the password
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    # Create new user
    new_user = User(full_name=full_name, email=email, password=hashed_password, otp=None)

    # Add the new user to the database
    db.session.add(new_user)
    db.session.commit()

    return redirect(url_for('login'))

# Login route
@app.route('/login_user', methods=['POST'])
def login_user():
    email = request.form['email']
    password = request.form['password']

    # Check if the user is admin
    if email == "admin@gmail.com":
        if password == "adminyolo":
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('login.html', login_error="Invalid admin password")

        # Check if the user exists
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return render_template('login.html', login_error="Invalid email or password")

    # Redirect to the dashboard after successful login


    # Store user information in the session
    session['user_id'] = user.id
    session['full_name'] = user.full_name
    session['username'] = user.email  # Use email as username
    session['login_time'] = datetime.now()

    # Redirect to the dashboard after successful login
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    # Fetch all detections with user relationship
    detections = db.session.query(
        Detection,
        User.full_name,
        User.email
    ).join(User, Detection.username == User.email)\
     .order_by(Detection.login_time.desc())\
     .all()

    files = []  # Placeholder for file listing, update as needed
    total_users = User.query.count()
    total_objects = Detection.query.count()
    total_contacts = ContactMessage.query.count()

    return render_template('admin_dashboard.html', 
                           detections=detections,
                           files=files,
                           total_users=total_users,
                           total_objects=total_objects,
                           total_contacts=total_contacts)

# Predict route for image processing
@app.route('/predict', methods=['POST'])
@login_required
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']

    # Generate a unique filename
    unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    # Save the input image
    file.save(file_path)

    # Read the saved image as bytes
    with open(file_path, "rb") as f:
        img_bytes = f.read()

    # Preprocess the image (Now img_bytes is correctly passed)
    img = preprocess_image(img_bytes)

    # Run the YOLOv8 model to get predictions
    results = model(img)
    predictions = results[0].boxes.data.tolist()

    # Draw bounding boxes and labels on the image
    img_with_detections, detected_classes = draw_detections_with_index(img, predictions)

    # Calculate confidence as the average confidence of detections
    if predictions:
        confidence = np.mean([pred[4] for pred in predictions])  # Average confidence
    else:
        confidence = 0.0

    # Save the result image
    output_image_path = 'static/output.jpg'
    cv2.imwrite(output_image_path, img_with_detections)

    # Save detected classes to the database
    detected_classes_str = ', '.join(detected_classes)
    new_detection = Detection(
        full_name=session['full_name'],
        username=session['username'],
        media_type='image',
        input_file_path=unique_filename,
        confidence=confidence,
        login_time=session['login_time'],
        logout_time=None,
        detected_classes=detected_classes_str,
        report=f"/user_activity/{session['username']}",  # Link to user activity
        satisfactory=None  # Feedback will be updated later
    )
    db.session.add(new_detection)
    db.session.commit()

    # In both predict routes, modify the return statement:
    return jsonify({
        "output_image": output_image_path,
        "detected_classes": detected_classes,
        "detection_id": new_detection.id  # Add this line
    })

@app.route('/user_activity/<username>')
def user_activity(username):
    # Fetch user details and recent detections
    user = User.query.filter_by(email=username).first()
    detections = Detection.query.filter_by(username=username)\
                               .order_by(Detection.login_time.desc())\
                               .limit(10)\
                               .all()
    
    return render_template('user_activity.html', 
                         detections=detections,
                         user=user,
                         username=username)

@app.route('/fetch_more_detections/<username>')
def fetch_more_detections(username):
    offset = request.args.get('offset', default=0, type=int)
    
    # Fetch the next 10 detections
    detections = Detection.query.filter_by(username=username)\
                               .order_by(Detection.login_time.desc())\
                               .offset(offset)\
                               .limit(10)\
                               .all()
    
    # Convert detections to a list of dictionaries
    detections_list = [
        {
            'full_name': detection.full_name,
            'media_type': detection.media_type,
            'confidence': detection.confidence,
            'login_time': detection.login_time.isoformat(),
            'logout_time': detection.logout_time.isoformat() if detection.logout_time else None,
            'detected_classes': detection.detected_classes,
            'satisfactory': detection.satisfactory
        }
        for detection in detections
    ]
    
    return jsonify(detections_list)

# Predict route for video processing
@app.route('/predict-video', methods=['POST'])
@login_required
def predict_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video provided"}), 400

    file = request.files['video']
    
    # Generate a unique filename for the uploaded video
    unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
    input_video_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    # Save the uploaded video
    file.save(input_video_path)

    # Open the saved video for processing
    cap = cv2.VideoCapture(input_video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_video_path = 'static/output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    all_detected_classes = set()  # To store unique detected classes
    all_confidences = []  # To store all confidence values

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        predictions = results[0].boxes.data.tolist()
        frame_with_detections, detected_classes = draw_detections_with_index(frame, predictions)

        all_detected_classes.update(detected_classes)  # Update the set of detected classes

        # Collect all confidence values for confidence
        all_confidences.extend([pred[4] for pred in predictions])
        out.write(frame_with_detections)

    cap.release()
    out.release()

    # Calculate average confidence for the video
    confidence = np.mean(all_confidences) if all_confidences else 0.0

    # Save detected classes to the database
    detected_classes_str = ', '.join(all_detected_classes)
    new_detection = Detection(
        full_name=session['full_name'],
        username=session['username'],
        media_type='video',
        input_file_path=unique_filename,
        confidence=confidence,
        login_time=session['login_time'],
        logout_time=None,
        detected_classes=detected_classes_str,
        report=f"/user_activity/{session['username']}",  # Link to user activity
        satisfactory=None  # Feedback will be updated later
    )
    db.session.add(new_detection)
    db.session.commit()

    # In both predict routes, modify the return statement:
    return jsonify({
        "output_video": output_video_path,
        "detected_classes": detected_classes,
        "detection_id": new_detection.id  # Add this line
    })   
    
# Logout route
@app.route('/logout')
@login_required
def logout():
    if 'user_id' in session:
        # Update the logout time in the database
        user_id = session['user_id']
        login_time = session['login_time']
        logout_time = datetime.now()

        # Fetch the latest detection record for the user
        detection = Detection.query.filter_by(username=session['username'], login_time=login_time).first()
        if detection:
            detection.logout_time = logout_time
            db.session.commit()

        # Clear the session
        session.clear()

    return redirect(url_for('home'))
@app.route('/stream', methods=['POST'])
def process_frame():
    try:
        # Parse the incoming JSON data
        data = request.json
        frame_data = data.get("frame")
        if not frame_data:
            return jsonify({"error": "No frame data provided"}), 400

        # Check if the frame has a valid Base64 prefix
        if not frame_data.startswith("data:image/jpeg;base64,"):
            print("Invalid frame format received:", frame_data[:50])
            return jsonify({"error": "Invalid frame format"}), 400

        # Remove the Base64 URI prefix and decode
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        if not frame_bytes:
            print("Decoded frame buffer is empty")
            return jsonify({"error": "Empty frame buffer"}), 400

        # Decode the frame to a NumPy array
        np_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
        if frame is None:
            print("Failed to decode frame")
            return jsonify({"error": "Failed to decode frame"}), 400

        # Perform inference with YOLOv8
        results = model(frame)  # Perform object detection
        annotated_frame = results[0].plot()  # Annotate frame with bounding boxes

        # Extract detected class names
        detected_classes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes.data:
                class_idx = int(box[-1])  # The class index is the last element in the box tensor
                detected_classes.append(model.names[class_idx])  # Map class index to class name

        # Encode the annotated frame to Base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')

        # Return the processed frame and detected classes
        return jsonify({
            "processed_frame": encoded_frame,
            "detected_classes": detected_classes
        })

    except Exception as e:
        print("Error processing frame:", e)
        return jsonify({"error": str(e)}), 500
# @app.route('/stream', methods=['POST'])
# def process_frame():
#     try:
#         # Parse the incoming JSON data
#         data = request.json
#         frame_data = data.get("frame")
#         if not frame_data:
#             return jsonify({"error": "No frame data provided"}), 400

#         # Check if the frame has a valid Base64 prefix
#         if not frame_data.startswith("data:image/jpeg;base64,"):
#             print("Invalid frame format received:", frame_data[:50])
#             return jsonify({"error": "Invalid frame format"}), 400

#         # Remove the Base64 URI prefix and decode
#         frame_bytes = base64.b64decode(frame_data.split(',')[1])
#         if not frame_bytes:
#             print("Decoded frame buffer is empty")
#             return jsonify({"error": "Empty frame buffer"}), 400

#         # Decode the frame to a NumPy array
#         np_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
#         frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
#         if frame is None:
#             print("Failed to decode frame")
#             return jsonify({"error": "Failed to decode frame"}), 400

#         # Perform inference with YOLOv8
#         results = model(frame)  # Perform object detection
#         annotated_frame = results[0].plot()  # Annotate frame with bounding boxes

#         # Extract detected class names
#         detected_classes = []
#         if results[0].boxes is not None and len(results[0].boxes) > 0:
#             for box in results[0].boxes.data:
#                 class_idx = int(box[-1])  # The class index is the last element in the box tensor
#                 detected_classes.append(model.names[class_idx])  # Map class index to class name

#         # Encode the annotated frame to Base64
#         _, buffer = cv2.imencode('.jpg', annotated_frame)
#         encoded_frame = base64.b64encode(buffer).decode('utf-8')

#         # Return the processed frame and detected classes
#         return jsonify({
#             "processed_frame": encoded_frame,
#             "detected_classes": detected_classes
#         })

#     except Exception as e:
#         print("Error processing frame:", e)
#         return jsonify({"error": str(e)}), 500
    
@app.route('/submit_satisfactory/<int:detection_id>', methods=['POST'])
@login_required
def submit_satisfactory(detection_id):
    try:
        data = request.json
        answer = data.get('answer')

        # Check if the detection exists
        detection = Detection.query.get(detection_id)
        if not detection:
            return jsonify({"error": "Detection not found"}), 404

        # Check if the user has already submitted feedback
        if detection.satisfactory is not None:
            return jsonify({"error": "Feedback already submitted"}), 400

        # Update the satisfactory field
        detection.satisfactory = answer
        db.session.commit()

        return jsonify({"success": True}), 200

    except Exception as e:
        print("Error in submit_satisfactory:", e)
        return jsonify({"error": str(e)}), 500
    
@app.route('/report/<username>')
def report(username):
    # Fetch user details
    user = User.query.filter_by(email=username).first()
    if not user:
        return render_template('404.html'), 404  # Handle case where user is not found

    # Fetch recent detections for the user
    detections = Detection.query.filter_by(username=username)\
                               .order_by(Detection.login_time.desc())\
                               .limit(10)\
                               .all()
    
    return render_template('report.html', user=user, detections=detections)
        
# Route to handle contact form submission
@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    username = request.form['username']
    email = request.form['email']
    message = request.form['message']

    new_message = ContactMessage(username=username, email=email, message=message)
    db.session.add(new_message)
    db.session.commit()

    return jsonify({"success": True, "message": "Message submitted successfully!"})
    
if __name__ == '__main__':
    app.run(debug=True)
