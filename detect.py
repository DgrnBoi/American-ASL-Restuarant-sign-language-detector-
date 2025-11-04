# detect.py (USE THIS FOR GITHUB)
from ultralytics import YOLO
import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="ASL Restaurant Sign Detector")
    parser.add_argument('--source', type=str, default='0', help='0 for webcam, or path to video/image')
    parser.add_argument('--conf', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--save', action='store_true', help='save output')
    parser.add_argument('--model', type=str, default='best.pt', help='path to model')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        return

    model = YOLO(args.model)
    print(f"Model loaded: {args.model}")

    source = args.source
    is_webcam = source == '0' or source.isdigit()

    if is_webcam:
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            print("ERROR: Cannot open webcam.")
            return
        print("WEBCAM LIVE | Press 'q' to quit")
    else:
        if not os.path.exists(source):
            print(f"ERROR: File not found: {source}")
            return
        print(f"Processing: {source}")

    out = None
    if args.save and is_webcam:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    print("\n" + "="*60)
    print("   AMERICAN ASL RESTAURANT SIGN DETECTOR")
    print("   PRESS 'q' TO QUIT")
    print("="*60 + "\n")

    while True:
        if is_webcam:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
        else:
            break

        results = model(frame if is_webcam else source, conf=args.conf, imgsz=args.imgsz, stream=True)
        for result in results:
            annotated = result.plot()
            cv2.imshow('ASL Detector', annotated)
            if out:
                out.write(annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if is_webcam:
                    cap.release()
                cv2.destroyAllWindows()
                if out:
                    out.release()
                return

    if not is_webcam:
        results = model(source, conf=args.conf, imgsz=args.imgsz, save=args.save)
        for r in results:
            cv2.imshow('Result', r.plot())
            cv2.waitKey(0