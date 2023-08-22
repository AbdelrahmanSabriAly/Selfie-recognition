import os
import base64
import cv2

COSINE_THRESHOLD = 0.5

import pickle

with open('2025_data.pkl', 'rb') as f:
    dictionary = pickle.load(f)

# Init models face detection & recognition
weights = os.path.join( "models",
                        "face_detection_yunet_2023mar_int8.onnx")
face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
face_detector.setScoreThreshold(0.87)

weights = os.path.join( "models", "face_recognition_sface_2021dec_int8.onnx")
face_recognizer = cv2.FaceRecognizerSF_create(weights, "")


def match( feature1):
    max_score = 0.0
    sim_user_id = ""
    for user_id, feature2 in zip(dictionary.keys(), dictionary.values()):
        score = face_recognizer.match(
            feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= max_score:
            max_score = score
            sim_user_id = user_id
    if max_score < COSINE_THRESHOLD:
        return False, ("", 0.0)
    return True, (sim_user_id, max_score)

def recognize_face(image,   file_name=None):
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.shape[0] > 1000:
        image = cv2.resize(image, (0, 0),
                           fx=500 / image.shape[0], fy=500 / image.shape[0])

    height, width, _ = image.shape
    face_detector.setInputSize((width, height))
    try:
        _, faces = face_detector.detect(image)
        if file_name is not None:
            assert len(faces) > 0, f'the file {file_name} has no face'

        faces = faces if faces is not None else []
        features = []
        for face in faces:

            aligned_face = face_recognizer.alignCrop(image, face)
            feat = face_recognizer.feature(aligned_face)

            features.append(feat)
        return features, faces
    except Exception as e:
        print(e)
        print(file_name)
        return None, None


def Process_Frames(image):
    # name_list = []
    fetures, faces = recognize_face(image)

    for idx, (face, feature) in enumerate(zip(faces, fetures)):
        result, user = match(feature)
        # box = list(map(int, face[:4]))
        # color = (0, 255, 0) if result else (0, 0, 255)
        # thickness = 2
        # cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

        id_name, score = user if result else (f"unknown_{idx}", 0.0)
        # text = "{0} ({1:.2f})".format(id_name, score)
        # position = (box[0], box[1] - 10)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # scale = 0.6
        # cv2.putText(image, text, position, font, scale,
        #             color, thickness, cv2.LINE_AA)
        print(id_name)
        # name_list.append(text)
    

    # # Encode the image to JPEG format and then to Base64
    # _, processed_image_data = cv2.imencode('.jpg', image)
    # processed_image_base64 = base64.b64encode(processed_image_data).decode('utf-8')

    return id_name


# def Process_Frames(image):

#     fetures, faces = recognize_face(image)

#     for idx, (face, feature) in enumerate(zip(faces, fetures)):
#         result, user = match( feature)
#         box = list(map(int, face[:4]))
#         color = (0, 255, 0) if result else (0, 0, 255)
#         thickness = 2
#         cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

#         id_name, score = user if result else (f"unknown_{idx}", 0.0)
#         text = "{0} ({1:.2f})".format(id_name, score)
#         position = (box[0], box[1] - 10)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         scale = 0.6
#         print(text)
#         cv2.putText(image, text, position, font, scale,
#                     color, thickness, cv2.LINE_AA)

#     # Write the frame with rectangles and recognized names to the output video
#     cv2.imshow("face recognition", image)
    

#     return image