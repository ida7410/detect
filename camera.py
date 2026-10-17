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


def look(frame):
    """Run both AI models on one frame. Returns everything the app needs."""
    h, w = frame.shape[:2]

    result = depth_model(frame, imgsz=config.IMG_SIZE, verbose=False)[0]
    depth = cv2.resize(result.depth.data.cpu().numpy(), (w, h)) * config.SCALE

    x1, y1, x2, y2 = cane_box(w, h)
    dist = float(np.percentile(depth[y1:y2, x1:x2], 10))

    det = detector(frame, imgsz=config.IMG_SIZE, verbose=False)[0]
    boxes = read_boxes(det)
    label = object_in_box(boxes, depth, x1, y1, x2, y2)

    return {"dist": dist, "label": label, "boxes": boxes, "box": (x1, y1, x2, y2)}