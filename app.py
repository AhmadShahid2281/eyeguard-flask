import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # Cross-Origin Resource Sharing (CORS) to allow your app to make requests

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB


# Mediapipe Face and Landmark Model
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


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

    # Initialize mediapipe face detection model
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return jsonify({'error': 'No face detected'}), 400

        # If face detected, proceed with analyzing eyes
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Crop the face from the image
            face_img = img[y:y + h, x:x + w]

            # Now, using Mediapipe's Face Mesh to detect the eyes
            with mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2) as face_mesh:
                face_mesh_results = face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

                if not face_mesh_results.multi_face_landmarks:
                    return jsonify({'error': 'No eye landmarks detected'}), 400

                for landmarks in face_mesh_results.multi_face_landmarks:
                    # Landmarks for left and right eyes can be identified by specific indices (left_eye_indices, right_eye_indices)
                    left_eye_indices = [33, 133, 159, 145, 153, 144, 163, 33]  # indices for left eye landmarks
                    right_eye_indices = [362, 382, 386, 374, 373, 380, 384, 362]  # indices for right eye landmarks

                    # Let's calculate redness based on the red color intensity of eye regions
                    left_eye_region = [landmarks.landmark[i] for i in left_eye_indices]
                    right_eye_region = [landmarks.landmark[i] for i in right_eye_indices]

                    # Calculate average redness score for the eye region
                    redness_score = calculate_redness_score(left_eye_region, right_eye_region, face_img)

                    # Determine rest/work message based on redness score
                    result = "Take rest" if redness_score > 80 else "Continue work"

                    return jsonify({
                        "result": result,
                        "score": float(redness_score)
                    })

    return jsonify({'error': 'No eyes detected'}), 400


def calculate_redness_score(left_eye_region, right_eye_region, face_img):
    # Simple approach: Sum of red channel pixel values within the eye region
    red_channel = face_img[:, :, 2]
    left_eye_redness = np.mean([red_channel[int(landmark.y * face_img.shape[0]), int(landmark.x * face_img.shape[1])] for landmark in left_eye_region])
    right_eye_redness = np.mean([red_channel[int(landmark.y * face_img.shape[0]), int(landmark.x * face_img.shape[1])] for landmark in right_eye_region])
    
    # Return combined redness score from both eyes
    return (left_eye_redness + right_eye_redness) / 2


if __name__ == '__main__':
    # Use PORT from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
