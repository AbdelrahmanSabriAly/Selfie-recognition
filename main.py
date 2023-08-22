from flask import Flask, render_template, request, jsonify
import cv2
import base64
import numpy as np

from FaceRecog import Process_Frames
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/selfie')
def selfie():
    return render_template('face-recognition.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        image = request.files['image'].read()
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Replace this with your actual recognition result
        label = Process_Frames(img)

        return jsonify({'label': label})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
