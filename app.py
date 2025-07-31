from __future__ import division, print_function
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for,session,flash
from datetime import datetime
from flask_cors import CORS
import csv

app = Flask(__name__)
CORS(app)
app.secret_key = 'supersecretkey'  # Required for session management

# Load model and face cascade
model = tf.keras.models.load_model('emotion_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

GR_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')

video_folder = 'videos'
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Initialize video writer object
video_writer = None

detected_emotion = None

def detect_emotion(frame):
    global detected_emotion
    faces = face_cascade.detectMultiScale(frame, 1.05, 5)
    output = []

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_face, (48, 48))
        reshaped = resized.reshape(1, 48, 48, 1) / 255.0
        predictions = model.predict(reshaped)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]
        output.append(predicted_emotion)

        cv2.rectangle(frame, (x, y), (x + w, y + h), GR_dict[1], 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), GR_dict[1], -1)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if output:
        detected_emotion = max(set(output), key=output.count)

    return frame

def generate_frames():
    global video_writer, detected_emotion
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_emotion(frame)

            # Write the frame to the video writer
            if video_writer is not None:
                video_writer.write(frame)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_emotion')
def detect_emotion_route():
    global detected_emotion
    if detected_emotion is None:
        return jsonify({'emotion': 'no face detected'})
    stop_recording()
    emotion = detected_emotion
    detected_emotion = None  # Reset detected emotion after detection
    return jsonify({'emotion': emotion})

@app.route('/start_recording')
def start_recording():
    global video_writer
    now = datetime.now()
    video_filename = os.path.join(video_folder, f"{now.strftime('%Y%m%d_%H%M%S')}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
    return jsonify({'message': 'Recording started'})

@app.route('/stop_recording')
def stop_recording():
    global video_writer, cap
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    if cap is not None:
        cap.release()
        cap.open(0)  # Re-open the camera for future use
    return jsonify({'message': 'Recording stopped'})




'''

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['user']
        password = request.form['password']
        with open('login.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == username and row[1] == password:
                    session['username'] = username
                    return redirect(url_for('home'))
        flash('Invalid credentials', 'error')
    return render_template('login.html')
'''

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']
        # Check if username (email) already exists
        with open('login.csv', 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == username:
                    flash('Email already exists', 'error')
                    return redirect(url_for('signup'))
        # If username does not exist, add new user to CSV
        with open('login.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([username, password])
        flash('Signup successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']
        with open('login.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row if exists
            user_found = False
            for row in reader:
                if row[0] == username:
                    if row[1] == password:
                        session['username'] = username
                        return redirect(url_for('home'))
                    else:
                        flash('Password incorrect', 'error')
                        user_found = True
                        break
                else:
                    continue
            if not user_found:
                flash('Username incorrect', 'error')
    return render_template('login.html')










@app.route('/home')
def home():
    if 'username' in session:
        username = session['username']
        return render_template('index1.html', username=username)
    return redirect(url_for('login'))

'''
@app.route("/")
def home():
    return render_template("index1.html")

'''
@app.route('/camera', methods=['GET', 'POST'])
def camera():
    return render_template("emotion_detection.html")

@app.route('/show_buttons')
def show_buttons():
    emotion = request.args.get('emotion')
    return render_template("buttons.html", final_output=emotion)

@app.route('/movies/surprise', methods=['GET', 'POST'])
def moviesSurprise():
    return render_template("moviesSurprise.html")

@app.route('/movies/angry', methods=['GET', 'POST'])
def moviesAngry():
    return render_template("moviesAngry.html")

@app.route('/movies/sad', methods=['GET', 'POST'])
def moviesSad():
    return render_template("moviesSad.html")

@app.route('/movies/disgust', methods=['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesSad.html")

@app.route('/movies/happy', methods=['GET', 'POST'])
def moviesHappy():
    return render_template("moviesHappy.html")

@app.route('/movies/fear', methods=['GET', 'POST'])
def moviesFear():
    return render_template("moviesFear.html")

@app.route('/movies/neutral', methods=['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesNeutral.html")

@app.route('/songs/surprise', methods=['GET', 'POST'])
def songsSurprise():
    return render_template("songsNeutral.html")

@app.route('/songs/angry', methods=['GET', 'POST'])
def songsAngry():
    return render_template("songsSoothing.html")

@app.route('/songs/sad', methods=['GET', 'POST'])
def songsSad():
    return render_template("songsSad.html")

@app.route('/songs/disgust', methods=['GET', 'POST'])
def songsDisgust():
    return render_template("songsSad.html")

@app.route('/songs/happy', methods=['GET', 'POST'])
def songsHappy():
    return render_template("songsHappy.html")

@app.route('/songs/fear', methods=['GET', 'POST'])
def songsFear():
    return render_template("songsFear.html")

@app.route('/songs/neutral', methods=['GET', 'POST'])
def songsNeutral():
    return render_template("songsNeutral.html")

@app.route('/templates/join_page', methods=['GET', 'POST'])
def join():
    return render_template("join_page.html")

@app.route('/templates/contact', methods=['GET', 'POST'])
def contact():
    return render_template("contact.html")

@app.route('/templates/login', methods=['GET', 'POST'])
def logins():
    return render_template("login.html")




if __name__ == "__main__":
    app.run(debug=True)
