import cv2
import config
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_boxes(img, boxes):
    """Every detected box: real name in blue, 'object' in gray."""
    for bx1, by1, bx2, by2, name in boxes:
        if name:
            text, color = name, (255, 128, 0)
        else:
            text, color = "object", (160, 160, 160)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(img, text, (bx1, max(by1 - 5, 15)), FONT, 0.6, color, 2)


def draw_info(img, info, video_fps, ai_fps):
    """Green box, orange side boxes, and the status line."""
    if info is None:
        text = "starting..."
    else:
        x1, y1, x2, y2 = info["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # orange box on a side when something is close there
        w = img.shape[1]
        for side, sx1, sx2 in (("left", 0, x1), ("right", x2, w)):
            thing = info[side]
            if thing:
                cv2.rectangle(img, (sx1, y1), (sx2, y2), (0, 165, 255), 2)
                cv2.putText(img, thing, (sx1 + 10, y1 + 30), FONT, 0.7, (0, 165, 255), 2)

        text = f"{info['label'] or 'object'}: {info['dist']:.2f} m  edges {info['edges']:.0%}"
    cv2.putText(img, f"{text}   video {video_fps:.0f} / AI {ai_fps:.1f} FPS",
                (10, 30), FONT, 0.7, (0, 255, 0), 2)


def draw_heights(img, heights, depth):
    """Tint pixels: green = floor, yellow = low obstacle, red = below the floor"""
    near = depth < config.LOW_RANGE # only color things lower range
    overlay = img.copy()
    overlay[near & (np.abs(heights) < config.FLOOR_TOL)] = (0, 255, 0) # floor
    overlay[near & (heights >= config.FLOOR_TOL) & (heights < config.LOW_TOP)] = (0, 255, 255) # low obstacle
    overlay[near & (heights < -config.FLOOR_TOL)] = (0, 0, 255) # below surface
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)
