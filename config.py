DEPTH_MODEL = "yolo26n-depth.pt"
DETECT_MODEL = "yolo26n.pt"
IMG_SIZE = 320

SCALE = 0.53
BOX_W = 0.5
BOX_H = 0.5

IMPORTANT = {"person", "bicycle", "car", "motorcycle", "bus", "truck",
             "chair", "bench", "couch", "dining table", "dog",
             "fire hydrant", "stop sign", "potted plant", "suitcase"}
MIN_CONF = 0.5

SPEAK_RANGE = 3.0
REPEAT_EVERY = 4.0
VERY_CLOSE = 1.0

EDGE_STEP = 8 # compare each pixel with its right n pixels
EDGE_JUMP = 0.3 # distance jumps by n%+ = a "cliff" (an object's edge)
EDGE_ROWS = 0.2 # wall if n% or fewer of the rows contain a cliff
SIDE_RANGE = 1.0


# --- Camera position (for height math) ---
# TODO(phone): CAM_HEIGHT -> user sets once in settings
# TODO(phone): CAM_TILT   -> read live from the phone's motion sensor
# TODO(phone): CAM_VFOV   -> read from the phone camera's specs
CAM_HEIGHT = 1.00 # meters from the floor to the webcam
CAM_TILT = 0.0 # degrees pointing down (0 = looking straight ahead)
CAM_VFOV = 45.0 # degrees the camera sees from top to bottom
LOW_RANGE = 3.0 # only care about low obstacles / drop-offs closer than this (meters)

FLOOR_TOL = 0.15 # meters: within ±n cm of 0 counts as floor
LOW_TOP = 1.0 # meters: low obstacles are between FLOOR_TOL and this height
LOW_MIN = 0.03 # at least 3% of the path area must be hazard pixels to count