import time
import cv2
import numpy as np
from ultralytics import YOLO

from rules import Announcer
from voice import speak, is_speaking, stop

model = YOLO("yolo26n-depth.pt")
detector = YOLO("yolo26n.pt")

announcer = Announcer()
SCALE = 0.53
BOX_W = 0.5   # box width  = half of the screen
BOX_H = 0.5   # box height = half of the screen

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Try changing 0 to 1.")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # NEW 1: don't let old frames pile up

IMPORTANT = {"person", "bicycle", "car", "motorcycle", "bus", "truck",
             "chair", "bench", "couch", "dining table", "dog",
             "fire hydrant", "stop sign", "potted plant", "suitcase"}
MIN_CONF = 0.5 # ignore guesses less than 50% sure
IMPORTANT_IDS = [i for i, n in detector.names.items() if n in IMPORTANT]

def read_boxes(det):
    """Turn the detector's result into a simple list:
    [(x1, y1, x2, y2, name), ...]  name = real name, or None if unsure / not important"""
    boxes = []
    for (bx1, by1, bx2, by2), cls, conf in zip(det.boxes.xyxy.tolist(),
                                               det.boxes.cls.tolist(),
                                               det.boxes.conf.tolist()):
        name = det.names[int(cls)]
        if name not in IMPORTANT or conf < MIN_CONF:
            name = None                      # the ONE place this rule lives
        boxes.append((int(bx1), int(by1), int(bx2), int(by2), name))
    return boxes


def object_in_box(boxes, depth, x1, y1, x2, y2):
    """Name of the closest named object overlapping the green box, or None."""
    best_name, best_dist = None, 999
    for bx1, by1, bx2, by2, name in boxes:
        if name is None:
            continue                                         # unnamed: skip
        if not (bx1 < x2 and bx2 > x1 and by1 < y2 and by2 > y1):
            continue                                         # not in my path
        d = float(np.median(depth[by1:by2, bx1:bx2]))
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name


def draw_boxes(img, boxes):
    """Draw every box: real name in blue, 'object' in gray."""
    for bx1, by1, bx2, by2, name in boxes:
        if name:
            text, color = name, (255, 128, 0)        # blue
        else:
            text, color = "object", (160, 160, 160)  # gray
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(img, text, (bx1, max(by1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

prev = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model(frame, imgsz=320, verbose=False)[0] # verbose false: no text everytime
    depth = result.depth.data.cpu().numpy() # get depth grid data, 1 per pixel
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0])) * SCALE # resize to fit in cam
    h, w = depth.shape
    center = depth[h//2 - 20:h//2 + 20, w//2 - 20:w//2 + 20]
    
    bw, bh = int(w * BOX_W), int(h * BOX_H)       # box size in pixels
    x1, x2 = (w - bw) // 2, (w + bw) // 2          # left and right edges
    y1, y2 = (h - bh) // 2, (h + bh) // 2          # top and bottom edges
    dist = float(np.percentile(depth[y1:y2, x1:x2], 10))   # closest stuff in the box
    
    det = detector(frame, imgsz=320, verbose=False)[0]
    boxes = read_boxes(det)                              # clean it up once
    label = object_in_box(boxes, depth, x1, y1, x2, y2)

    now = time.time()
    message, interrupt = announcer.decide(dist, now, is_speaking())

    if message:
        if interrupt:
            stop()
        print(message)
        speak(message)

    # debug window
    vis = np.clip(depth, 0, 5)
    vis = (255 - vis / 5 * 255).astype(np.uint8) # convert depth to colour
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    fps = 1 / (now - prev)
    prev = now

    draw_boxes(frame, boxes) # draw the detector's boxes and names on the camera image
    for img in (frame, vis):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label or 'object'}: {dist:.2f} m   FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Detect | depth", np.hstack([frame, vis]))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()