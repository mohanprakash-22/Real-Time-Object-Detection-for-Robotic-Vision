import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;error"

import cv2
import numpy as np

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Camera selection with automatic /video handling
source = input("Enter camera source (0 for webcam or IP camera URL): ").strip()

# Automatically format IP camera URL
if "://" in source and "/video" not in source:
    # Split into protocol and rest
    protocol, rest = source.split("://", 1)
    # Check if there's a path component
    if "/" not in rest:
        source = f"{protocol}://{rest}/video"
    else:
        domain_port, path = rest.split("/", 1)
        if not path:  # Handle trailing slash
            source = f"{protocol}://{domain_port}/video"

if source == "0":
    source = 0  # Convert to integer for local camera

cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer for IP cameras

# Verify camera connection
if not cap.isOpened():
    print(f"Error: Could not open video source {source}")
    exit(1)

# Detection stabilization improvements
DETECTION_HISTORY = 3
MAX_HISTORY_LENGTH = 50
detection_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error receiving frame - check camera connection")
        break
    
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    # YOLO detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Current frame detection data
    curr_boxes = []
    curr_confidences = []
    curr_class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                curr_boxes.append([center_x - w//2, center_y - h//2, w, h])
                curr_confidences.append(float(confidence))
                curr_class_ids.append(class_id)

    # Maintain detection history
    detection_history.append({
        'boxes': curr_boxes,
        'confidences': curr_confidences,
        'class_ids': curr_class_ids
    })
    detection_history = detection_history[-DETECTION_HISTORY:]

    # Combine detections
    combined_boxes = []
    combined_confidences = []
    combined_class_ids = []
    
    for entry in detection_history:
        combined_boxes.extend(entry['boxes'])
        combined_confidences.extend(entry['confidences'])
        combined_class_ids.extend(entry['class_ids'])

    # Non-max suppression
    if len(combined_boxes) > 0:
        indexes = cv2.dnn.NMSBoxes(combined_boxes, combined_confidences, 0.5, 0.4)
        indexes = np.array(indexes).flatten()
    else:
        indexes = []

    # Draw detections
    for i in indexes:
        if i < len(combined_boxes):
            x, y, w, h = combined_boxes[i]
            label = f"{classes[combined_class_ids[i]]} {combined_confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Enhanced Robot Vision", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
