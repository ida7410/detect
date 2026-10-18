import cv2
import numpy as np
from ultralytics import YOLO

import config

depth_model = YOLO(config.DEPTH_MODEL)
detector = YOLO(config.DETECT_MODEL)


def read_boxes(det):
    """[(x1, y1, x2, y2, name), ...]  name = real name, or None if unsure / not important"""
    boxes = []
    for (bx1, by1, bx2, by2), cls, conf in zip(det.boxes.xyxy.tolist(),
                                               det.boxes.cls.tolist(),
                                               det.boxes.conf.tolist()):
        name = det.names[int(cls)]
        if name not in config.IMPORTANT or conf < config.MIN_CONF:
            name = None
        boxes.append((int(bx1), int(by1), int(bx2), int(by2), name))
    return boxes


def object_in_box(boxes, depth, x1, y1, x2, y2):
    """Name of the closest named object overlapping the cane box, or None."""
    best_name, best_dist = None, 999
    for bx1, by1, bx2, by2, name in boxes:
        if name is None:
            continue
        if not (bx1 < x2 and bx2 > x1 and by1 < y2 and by2 > y1):
            continue
        d = float(np.median(depth[by1:by2, bx1:bx2]))
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name


def cane_box(w, h):
    """The green box in the middle of the screen: (x1, y1, x2, y2)."""
    bw, bh = int(w * config.BOX_W), int(h * config.BOX_H)
    return (w - bw) // 2, (h - bh) // 2, (w + bw) // 2, (h + bh) // 2


def edge_rows(area):
    """Share of rows (0.0 to 1.0) that contains a 'cliff'"""
    k = config.EDGE_STEP
    here = area[:, :-k] # each pixel except right k pixles
    neighbor = area[:, k:] # each pixel after k pixels
    jump = np.abs(neighbor - here) / np.minimum(neighbor, here) # jump as %
    has_cliff = (jump > config.EDGE_JUMP).any(axis=1) # True for each row with a cliff
    return float(has_cliff.mean()) # share of rows with a cliff


def side_check(depth, boxes, sx1, y1, sx2, y2):
    """One side zone: a name 'wall' or None if nothing is close"""
    d = float(np.percentile(depth[y1:y2, sx1:sx2], 10))
    if d >= config.SIDE_RANGE:
        return None
    return object_in_box(boxes, depth, sx1, y1, sx2, y2) or "wall"


def height_map(depth):
    """Height above the floor(metres) for every pixel."""
    h, w = depth.shape

    # turn how many rows below the middle into angle
    f = (h / 2) / np.tan(np.radians(config.CAM_VFOV / 2)) # how many rows below the middle
    rows = np.arange(h).reshape(-1, 1) # rows 0, 1, ... one per row
    down = depth * (rows - h / 2) / f # how far below the camera's centre line
    # rows - h / 2 = each row's relative position to the middle, (-) above, (+) below

    tilt = np.radians(config.CAM_TILT) # add the tilt
    drop = down * np.cos(tilt) + depth * np.sin(tilt) # get total drop below camera

    return config.CAM_HEIGHT - drop # height = camera height - drop


def look(frame):
    """Run both AI models on one frame. Returns everything the app needs."""
    h, w = frame.shape[:2]

    result = depth_model(frame, imgsz=config.IMG_SIZE, verbose=False)[0]
    depth = cv2.resize(result.depth.data.cpu().numpy(), (w, h)) * config.SCALE
    heights = height_map(depth)

    x1, y1, x2, y2 = cane_box(w, h)
    region = depth[y1:y2, x1:x2]
    dist = float(np.percentile(region, 10))

    det = detector(frame, imgsz=config.IMG_SIZE, verbose=False)[0]
    boxes = read_boxes(det)
    label = object_in_box(boxes, depth, x1, y1, x2, y2)

    band = depth[y1:y2, :]
    edges = edge_rows(band)
    if label is None and edges <= config.EDGE_ROWS:
        label = "wall"

    left = side_check(depth, boxes, 0, y1, x1, y2)
    right = side_check(depth, boxes, x2, y1, w, y2)

    return {"dist": dist, "label": label, "boxes": boxes, "box": (x1, y1, x2, y2), 
            "edges": edges, "left": left, "right": right, "heights": heights, "depth": depth}