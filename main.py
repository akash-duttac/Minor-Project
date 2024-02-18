# modified to only detect persons and added a basic counter

import cv2
import argparse

from ultralytics import YOLO
import supervision as sv


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

        person_count = len(person_detections)
        height, width, _ = frame.shape
        text_x, text_y = 10, 20
        cv2.putText(frame, f"People: {person_count}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("yolov8 (Press 'esc' to close the window)", frame)
        # print(frame.shape)
        # break

        # if esc is pressed->stop loop
        if (cv2.waitKey(30) == 27): # 30ms wait & 27 - ASCII for escape
            break

if __name__ == "__main__":
    main()