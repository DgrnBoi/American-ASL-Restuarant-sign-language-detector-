# detect.py
# Real-time ASL Restaurant Sign Detector
# Supports: Webcam | Video | Image | Save Output

from ultralytics import YOLO
import cv2
import argparse
import os

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="ASL Restaurant Sign Detector")
    parser.add_argument('--source', type=str, default='0', help='0 for webcam, or path to video/image')
    parser.add_argument('--conf', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--save', action='store_true', help='save output video/image')
    parser.add_argument('--model', type=str, default='best.pt', help='path to model')
    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        return
    model = YOLO(args.model)
    print(f"Model loaded: {args.model}")

    # Handle source
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

    # Setup video writer if saving
    out = None
    if args.save and is_webcam:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    # Header
    print("\n" + "="*60)
    print("   AMERICAN ASL RESTAURANT SIGN DETECTOR")
    print("   PRESS 'q' TO QUIT")
    print("="*60 + "\n")

    # Webcam loop
    while is_webcam:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))

        results = model(frame, conf=args.conf, imgsz=args.imgsz, stream=True)
        for result in results:
            annotated = result.plot()
            cv2.imshow('ASL Detector - LIVE', annotated)
            if out:
                out.write(annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                if out:
                    out.release()
                return

    # Non-webcam (image/video)
    if not is_webcam:
        results = model(source, conf=args.conf, imgsz=args.imgsz, save=args.save)
        for r in results:
            annotated = r.plot()
            cv2.imshow('ASL Detection Result', annotated)
            cv2.waitKey(0)  # ← FIXED: Added missing )
        cv2.destroyAllWindows()

    print("Detection complete!")

if __name__ == '__main__':
    main()
