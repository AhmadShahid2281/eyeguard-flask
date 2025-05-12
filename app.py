from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load OpenCV's pre-trained Haar Cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 400

    # Loop through each face detected
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Extract the eye region from the image
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            # Analyze the redness of the eye
            red_channel = eye_roi[:, :, 2]
            redness_score = np.mean(red_channel)

            # Adjust threshold for redness detection
            result = "Take rest" if redness_score > 80 else "Continue work"
            return jsonify({"result": result, "score": float(redness_score)})

    return jsonify({"error": "No eyes detected"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
