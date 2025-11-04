from ultralytics import YOLO
import cv2

# LOAD YOUR MODEL
model = YOLO(r"C:\Users\Eshaan Sunthankar\Desktop\runs\asl_local\weights\best.pt")

# OPEN WEBCAM
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: WEBCAM NOT FOUND")
    exit()

print("\n" + "="*50)
print("   YOUR RESTAURANT ASL DETECTOR IS LIVE!")
print("   MAKE A SIGN: hello, menu, burger, please, thank-you")
print("   PRESS 'q' TO QUIT")
print("="*50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # DETECT
    results = model(frame, conf=0.05, imgsz=640)[0]
    annotated = results.plot()

    # SHOW
    cv2.imshow('YOUR ASL MODEL - LIVE', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("WEBCAM CLOSED. YOU ARE DONE.")
