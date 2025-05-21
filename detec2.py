
# -------------------------------------------------
# 0. auto-install any missing deps
# -------------------------------------------------
import sys, subprocess, warnings, logging, os, cv2
def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
for pkg in ("ultralytics==8.3.140", "deep-sort-realtime", "opencv-python"):
    try:
        __import__(pkg.split("-")[0])
    except ImportError:
        pip_install(pkg)

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# -------------------------------------------------
# 1. imports
# -------------------------------------------------
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# -------------------------------------------------
# 2. user-tunable knobs
# -------------------------------------------------
PERSON_MODEL = "yolov8n.pt"                          
TAG_MODEL    = r"runs/detect/train/weights/best.pt" 
TAG_CONF     = 0.04      
PERSON_CONF  = 0.30
IOU_THRESH   = 0.25
INFER_SIZE   = 640       
MAX_AGE      = 30        
N_INIT       = 3         

# -------------------------------------------------
# 3. pick a video file
# -------------------------------------------------
Tk().withdraw()
video_path = askopenfilename(
    title="Select a video file",
    filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")]
)
if not video_path:
    sys.exit("No video selected.")

# -------------------------------------------------
# 4. load models & tracker
# -------------------------------------------------
person_yolo = YOLO(PERSON_MODEL)
tag_yolo    = YOLO(TAG_MODEL)

tracker = DeepSort(
    max_age=MAX_AGE,
    n_init=N_INIT,
    nms_max_overlap=1.0,
    max_cosine_distance=0.4,
)

staff_id = None          

# -------------------------------------------------
# 5. video loop
# -------------------------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"Cannot open {video_path}")

frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1
    h, w = frame.shape[:2]

    # 5a. detect persons
    detections, person_boxes = [], []
    res = person_yolo(frame, conf=PERSON_CONF, classes=[0],
                      imgsz=INFER_SIZE, verbose=False)[0]
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        conf = float(b.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf))
        person_boxes.append([x1, y1, x2, y2])

    tracks = tracker.update_tracks(detections, frame=frame)

    # 5b. tag detection if staff not yet assigned
    tag_boxes = []
    if staff_id is None:
        for (x1, y1, x2, y2) in person_boxes:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            tags = tag_yolo(roi, conf=TAG_CONF,
                            imgsz=INFER_SIZE, verbose=False)[0]
            for tb in tags.boxes:
                tx1, ty1, tx2, ty2 = map(int, tb.xyxy[0])
                tag_boxes.append([tx1 + x1, ty1 + y1, tx2 + x1, ty2 + y1])

    # 5c. iterate tracks, assign staff_id, draw
    for trk in tracks:
        if not trk.is_confirmed():
            continue
        tid = trk.track_id
        x, y, w_box, h_box = map(int, trk.to_ltwh())   
        x2, y2 = x + w_box, y + h_box

        # assign staff_id if tag overlaps this track
        if staff_id is None and tag_boxes:
            for tx1, ty1, tx2, ty2 in tag_boxes:
                ix1, iy1 = max(x, tx1), max(y, ty1)
                ix2, iy2 = min(x2, tx2), min(y2, ty2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (w_box * h_box) + (tx2 - tx1) * (ty2 - ty1) - inter
                iou   = inter / union if union else 0
                if iou > IOU_THRESH:
                    staff_id = tid
                    cx, cy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
                    print(f"Frame {frame_no}: tag at ({cx},{cy}) → staff_id {tid}")
                    break

        # draw only staff
        if tid == staff_id:
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "STAFF", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # reset if staff track lost
    if staff_id is not None:
        alive_ids = {t.track_id for t in tracks if t.is_confirmed()}
        if staff_id not in alive_ids:
            staff_id = None

    cv2.imshow("Follow Staff – press q", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
