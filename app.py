import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)  

# Limit uploaded file size to 2MB
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB

# Load Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    # Convert image file to NumPy array
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    # Resize image to reduce memory usage
    img = cv2.resize(img, (640, 480))

    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 400

    # Loop through detected faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect eyes within face
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 0:
            continue  # Try next face if no eyes found

        for (ex, ey, ew, eh) in eyes:
            # Extract eye region
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]

            # Analyze redness from red channel
            red_channel = eye_roi[:, :, 2]
            redness_score = np.mean(red_channel)

            # Threshold for redness score
            result = "Take rest" if redness_score > 80 else "Continue work"

            return jsonify({
                "result": result,
                "score": float(redness_score)
            })

    return jsonify({'error': 'No eyes detected'}), 400

if __name__ == '__main__':
    # Use PORT from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
