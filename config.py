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