
# ---------- 1. Install / import ----------
import sys, subprocess, warnings, logging
for pkg in ["opencv-python", "torch", "torchvision"]:
    try: __import__(pkg.split('-')[0])
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import cv2, numpy as np, torch
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Silence YOLOv5 autocast deprecation spam
warnings.filterwarnings("ignore",
                        message="`torch.cuda.amp.autocast",
                        category=FutureWarning)
logging.getLogger("torch").setLevel(logging.ERROR)

# ---------- 2. Pick video ----------
Tk().withdraw()
video_path = askopenfilename(title="Select a video file",
                             filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"),
                                        ("All files", "*.*")])
if not video_path:
    print("No video selected – exiting.")
    sys.exit(0)

# ---------- 3. Load YOLOv5 ----------
print("Loading YOLOv5s…")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf, model.iou, model.classes = 0.25, 0.45, [0]  

# ---------- 4. Video loop ----------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open {video_path}")
    sys.exit(1)

frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_no += 1
    h, w = frame.shape[:2]

    results = model(frame, size=640)
    persons = results.pandas().xyxy[0]

    staff_found = False

    for _, det in persons.iterrows():
        x1,y1,x2,y2 = map(int, [det.xmin, det.ymin, det.xmax, det.ymax])
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w-1,x2),min(h-1,y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_area = roi.shape[0]*roi.shape[1]
        for c in cnts:
            a = cv2.contourArea(c)
            if not (roi_area*0.001 < a < roi_area*0.1): continue
            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) == 4:          
                staff_found = True
              
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, "STAFF", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                cx, cy = (x1+x2)//2, (y1+y2)//2
                print(f"Frame {frame_no}: Staff at ({cx}, {cy})")
                break
        if staff_found:
            break  

  
    cv2.imshow("Staff Detection – press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




