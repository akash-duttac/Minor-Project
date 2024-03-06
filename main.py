# modified to only persons and gender and added a basic counter

import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv
import cvlib as cv
from keras.preprocessing.image import img_to_array
from keras.models import load_model

model = load_model('gender_detection.model')
classes = ['male', 'female']

def gender_detection(face, frame):
    # loop through detected faces
    for idx, f in enumerate(face):
         
         #get corner points of facce rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        #draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        #crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        #preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis = 0)

        #apply gender detection on face
        conf = model.predict(face_crop)[0]
                # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        
        #get label with maximum accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx]*100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#to set resolution
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default = [1280, 720],
        nargs = 2,
        type = int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 2,
        text_scale = 1
    )

    while True:
        ret, frame = cap.read()

        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        
         # Filter out detections other than persons
        person_detections = []
        for detection in detections:
            _, confidence, class_id, _ = detection
            if model.model.names[class_id] == "person":
                person_detections.append(detection)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in person_detections
        ]
        
        frame = box_annotator.annotate(scene = frame, detections = person_detections, labels = labels)

        face, confidence = cv.detect_face(frame)

        gender_detection(face, frame)

        person_count = len(person_detections)
        height, width, _ = frame.shape
        text_x, text_y = 10, 20
        cv2.putText(frame, f"People: {person_count}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("yolov8 (Press 'esc' to close the window)", frame)
        # print(frame.shape)
        # break

        # if esc is pressed->stop loop
        if (cv2.waitKey(30) == 27): # 30ms wait & 27 - ASCII for escape
            break

if __name__ == "__main__":
    main()
