import threading
import time
import cv2

from camera import look
from rules import Announcer
from voice import speak, is_speaking, stop
from draw import draw_boxes, draw_info

announcer = Announcer()

# share mailbox
lock = threading.Lock()
latest_frame = None
latest_info = None
ai_fps = 0.0
running = True

def ai_worker():
    """🧠 Background worker: eyes -> brain -> mouth, as fast as the AI allows."""
    global latest_info, ai_fps
    prev = time.time()
    while running:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        info = look(frame) # camera
        message, interrupt = announcer.decide( # should it describe?
            info["dist"], time.time(), is_speaking(), info["label"])
        if message: # voice
            if interrupt:
                stop()
            print(message)
            speak(message)

        now = time.time()
        with lock:
            latest_info = info
            ai_fps = 1 / (now - prev)
        prev = now

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Try changing 0 to 1.")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # NEW 1: don't let old frames pile up

threading.Thread(target=ai_worker, daemon=True).start()

# video loop
prev = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break

    with lock:
        latest_frame = frame
        info, fps_ai = latest_info, ai_fps

    # get fps
    now = time.time()
    video_fps = 1 / (now - prev)
    prev = now

    shown = frame.copy()
    if info:
        draw_boxes(shown, info["boxes"])
    draw_info(shown, info, video_fps, fps_ai)

    cv2.imshow("SafeStep", shown)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()