import os
import cv2
from flask import Flask, render_template, request
from deepface import DeepFace
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return 'No image file uploaded', 400
    
    file = request.files['image']
    
    # validate image size and shape
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return 'Invalid image file', 400
    
    # validate face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return 'No face detected in the image', 400
    
    img_path = './temp_img.jpg'
    cv2.imwrite(img_path, img)
    
    result = DeepFace.analyze(img_path, actions=['emotion'])
    
    # delete temporary image file
    os.remove(img_path)
    
    return result[0]['dominant_emotion'], 200


if __name__ == '__main__':
    app.run(port=5000, debug=True)
# run the application in debug mode
