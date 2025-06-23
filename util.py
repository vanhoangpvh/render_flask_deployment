import base64
import pickle

import joblib
import json
import numpy as np
import cv2
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image_b64_data, file_path = None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_b64_data)
    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        len_image_array = 32*32*3+32*32

        final = combined_img.reshape(1, len_image_array).astype(float)

        result.append({'Info': class_number_to_name(__model.predict(final)[0]),
                       'Probability': np.round(__model.predict_proba(final)*100, 2).tolist()[0],
                       'Class_dictionary': __class_name_to_number})

    return result

def load_saved_artifacts():
    print('Loading saved artifacts...Start')
    global __class_name_to_number
    global __class_number_to_name

    with open('./artifacts/class_dictionary.json', 'r') as f:
        player_info = json.load(f)

        __class_name_to_number = {name: info['label'] for name, info in player_info.items()}
        __class_number_to_name = {info['label']: name for name, info in player_info.items()}

    global __model
    global __player_info
    __player_info = player_info
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print('Loading saved artifacts...Done')

def class_number_to_name(class_num):
    name = __class_number_to_name[class_num]
    if name is None:
        return None  # Không tìm thấy

    info = __player_info[name]
    return {
        "name": name,
        "shirt_number": info["shirt_number"],
        "goals": info["goals"]
    }


def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_b64_data):
    face_cascade = cv2.CascadeClassifier('./opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv-4.x/data/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_b64_data)

    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    cropped_faces = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x: x+w]
        roi_color = img[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_test_image_for_ronaldo():
    with open('b64.txt') as f:
        return f.read()


if __name__ == '__main__':
    load_saved_artifacts()
    # print(classify_image(get_b64_test_image_for_ronaldo(), None))
    # print(classify_image(None, './test_images/Pele6.png'))
    # print(classify_image(None, './test_images/test3.jpg'))
    print(classify_image(None, './test_images/test4.png'))
    print(classify_image(None, './test_images/test7.png'))
