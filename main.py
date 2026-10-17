import time
import cv2
import numpy as np
from ultralytics import YOLO

from rules import Announcer
from voice import speak, is_speaking, stop

model = YOLO("yolo26n-depth.pt")
announcer = Announcer()
SCALE = 0.53
BOX_W = 0.5   # box width  = half of the screen
BOX_H = 0.5   # box height = half of the screen

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Try changing 0 to 1.")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # NEW 1: don't let old frames pile up

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

    for img in (frame, vis):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Center: {dist:.2f} m   FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Detect | depth", np.hstack([frame, vis]))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()