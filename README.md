<div align="center">
  <h1>American-ASL-Restaurant-Sign-Language-Detector</h1>
  <p><strong>Real-time ASL sign detector for restaurant ordering using YOLOv8</strong></p>
  <p>
    <span style="background:#06d6a0;color:white;padding:6px 12px;border-radius:20px;font-size:0.8rem;font-weight:bold;margin:5px;">YOLOv8</span>
    <span style="background:#06d6a0;color:white;padding:6px 12px;border-radius:20px;font-size:0.8rem;font-weight:bold;margin:5px;">Accessibility</span>
    <span style="background:#06d6a0;color:white;padding:6px 12px;border-radius:20px;font-size:0.8rem;font-weight:bold;margin:5px;">Open Source</span>
    <span style="background:#06d6a0;color:white;padding:6px 12px;border-radius:20px;font-size:0.8rem;font-weight:bold;margin:5px;">80+ Signs</span>
  </p>
</div>

---

## Overview

This project uses **Ultralytics YOLOv8** to detect **American Sign Language (ASL)** gestures in restaurant contexts — from ordering `pizza` to asking for the `bill`.

**Perfect for:**
- Accessibility apps in restaurants
- Sign-language kiosks
- ASL education tools
- AI/ML portfolio projects

---

## Detected Signs (80 Classes)

- alcohol, allergy, bacon, bag, barbecue, bill, biscuit, bitter, bread, burger
- bye, cake, cash, cheese, chicken, coke, cold, cost, coupon, credit card
- cup, dessert, drink, drive, eat, eggs, enjoy, fork, french fries, fresh
- hello, hot, icecream, ingredients, juicy, ketchup, lactose, lettuce, lid
- manager, menu, milk, mustard, napkin, 'no', order, pepper, pickle, pizza
- please, ready, receipt, refill, repeat, safe, salt, sandwich, sauce, small
- soda, sorry, spicy, spoon, straw, sugar, sweet, thank-you, tissues, tomato
- total, urgent, vegetables, wait, warm, water, what, would, yoghurt, your
- 
Total classes (nc): 80
Dataset & Credits

This project does NOT include the dataset. All data belongs to original authors.

Source: Roboflow Universe
roboflow:
  license: CC BY 4.0
  project: asl-72izb-fbciy
  url: https://universe.roboflow.com/asl-signs-from-alphabets-numbers-and-few-words/asl-72izb-fbciy/dataset/1
  version: 1
  workspace: asl-signs-from-alphabets-numbers-and-few-words
  
  Huge Thanks to the ASL Signs from Alphabets, Numbers and Few Words team.
Used Roboflow for augmentation, splitting, and YOLOv8 export.

Quick Start:
# 1. Clone the repo
git clone https://github.com/DgrnBoi/American-ASL-Restaurant-Sign-Language-Detector.git
cd American-ASL-Restaurant-Sign-Language-Detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run live webcam detection
python detect.py --source 0

Press q to quit.


Installation(on your command prompt):
# Python 3.8+ recommended
pip install ultralytics opencv-python torch torchvision
or 
pip install -r requirements.txt

Project Structure
American-ASL-Restaurant-Sign-Language-Detector/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data.yaml
├── best.pt
├── detect.py              # Pro version
├── webcam_demo.py         # Beginner version
└── demo/
    └── README.md
Note: Dataset and demo files are not included due to size and licensing.

Difficulties Faced:

Class imbalance: pizza, drink had more samples than yoghurt, lid
Hand occlusion & background noise in real restaurants
Lighting variations (indoor vs outdoor)
Low-resolution hands from distant webcams
No GPU: Training took ~6 hours on CPU
Annotation errors in original dataset

Solutions Applied:

Mosaic + HSV augmentation
Per-class confidence thresholds
Used yolov8n for speed
Post-processing filters


Contribute
Found a bug? Want to add signs?
Pull requests welcome!

Fork the repo
Create your feature branch
Commit changes
Open a PR with clear description

License
Code: MIT License © @DgrnBoi
Dataset: CC BY 4.0 – belongs to original Roboflow authors

Author:

  Eshaan Sunthankar

  Zeal College of Engineering | AIML Student

  GitHub: @DgrnBoi | LinkedIn: https://www.linkedin.com/in/eshaan-sunthankar-265054327/

  "Making restaurants accessible, one sign at a time."
— Open Source for Good
