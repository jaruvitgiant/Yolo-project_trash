from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
ELEPHANT_CLASS_ID = 20
ele = cv2.VideoCapture("ele.mp4")

while ele.isOpened():
    ret, frame = ele.read()
    if not ret:
        break
    results = model(frame, conf=0.3)  # ปรับ confidence ได้ตามต้องการ

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])  # class ID
            conf = box.conf[0]     # confidence
            if cls == ELEPHANT_CLASS_ID:
                # ตำแหน่งของกล่อง (bounding box)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"Elephant {conf:.2f}"
                
                # วาดกรอบและชื่อ
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
    cv2.imshow("Elephant Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # กด 'q' เพื่อหยุด
        break

# ปิดทุกอย่าง
ele.release()
cv2.destroyAllWindows()