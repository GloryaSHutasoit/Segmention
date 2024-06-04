from ultralytics import YOLO
import random
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(94, 480)

model = YOLO("terbaik.pt")

yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

conf = 0.128  

colors = [random.choices(range(256), k=3) for _ in classes_ids]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()
    alpha = 0.4 

    results = model.predict(frame, conf=conf)

    for result in results:
        if result.masks is not None and result.boxes is not None:
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                color_number = classes_ids.index(int(box.cls[0]))

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[color_number], 2)

                cv2.polylines(frame, [points], True, colors[color_number], 2)
                cv2.fillPoly(overlay, [points], colors[color_number])
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                label = f"{yolo_classes[int(box.cls[0])]}: {box.conf[0]:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), colors[color_number], cv2.FILLED)
                cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    cv2.imshow("Real-Time YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
