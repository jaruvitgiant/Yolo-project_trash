from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
cap = cv2.VideoCapture("video-test/tt.mp4")
if not cap.isOpened():
    print("ไม่สามารถเปิดวิดีโอได้")
TARGET_CLASS_IDS = [0, 1]  # IDs of classes to detect (e.g., plastic, glass)

# ชื่อคลาส (แก้ตาม data.yaml ที่คุณใช้เทรน)
CLASS_NAMES = ['plastic', 'glass',]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("อ่านเฟรมไม่ได้")

    # รัน YOLO บน frame
    results = model(frame, conf=0.3)

    # วาดผลลัพธ์
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id in TARGET_CLASS_IDS:
                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # แสดงภาพ
    cv2.imshow("Waste Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิด
cap.release()
cv2.destroyAllWindows()