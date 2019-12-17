import pickle
import io
import cv2
from imageio import imread
from threading import Thread

import cv2
import face_recognition
import flask
import base64

import requests
from flask import request, jsonify

app = flask.Flask(__name__)
encodings = "encodings.pickle"
data = pickle.loads(open(encodings, "rb").read())
detection_method = "hog"
test_image = "anupam.JPG"

@app.route('/api/recognize', methods = ['GET', 'POST'])
def recognise():
    body = request.get_json()
    encoded_image_str = body['content']

    # reconstruct image as an numpy array
    img = imread(io.BytesIO(base64.b64decode(encoded_image_str)))
    # print(type(img))
    name,bbox = recognise_image(img)
    result = {"name":name}
    return jsonify(result)

def recognise_image(image):
    # image = cv2.imread(test_image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb,
                                            model=detection_method)
    if not boxes:
        return "None",None
    encodings = face_recognition.face_encodings(rgb, boxes)
    if not encodings:
        return "None",None

    # initialize the list of names for each face detected
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)
        # print(names)

        return names[0],boxes

def recognize_video():
    cap = cv2.VideoCapture(-1)

    while True:
        ret,image = cap.read()

        if ret:
            recognised_name,boxes = recognise_image(image)
            if boxes is not None:
                (top, right, bottom, left) = boxes[0]
                # draw the predicted face name on the image
                cv2.rectangle(image, (left, top), (right, bottom),
                              (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(image, recognised_name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)

            print(recognised_name)
            cv2.imshow("frame",image)
            cv2.waitKey(1)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)