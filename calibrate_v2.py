import cv2
import yaml
import argparse
import shutil
import os
from datetime import datetime
import numpy as np

# ---------------- utils ----------------
def backup_settings(path="config/settings.yaml"):
    if os.path.exists(path):
        dst = f"{path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(path, dst)
        print(f"[INFO] Backup created: {dst}")

def draw_text(img, text, pos, scale=0.8, color=(255,255,255)):
    cv2.putText(img, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

def draw_mini_status(img, cam, lane, current_line, points):
    x, y = 20, 20
    pad = 10
    lines = [
        f"CAM : {cam.get('id','?')}",
        f"LANE: {lane.get('id','?')}",
        f"MODE: {current_line.upper()}",
        f"PT  : {len(points)+1} / 2"
    ]

    font_scale = 0.75
    line_h = 28
    box_w = 260
    box_h = line_h * len(lines) + pad * 2

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+box_w, y+box_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    yy = y + pad + line_h
    for txt in lines:
        draw_text(img, txt, (x+pad, yy), font_scale)
        yy += line_h

# ---------------- capture once ----------------
def capture_single_frame(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame

# ---------------- main ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--cam", type=int, default=0)
parser.add_argument("--lane", type=int, default=0)
args = parser.parse_args()

with open("config/settings.yaml", "r") as f:
    cfg = yaml.safe_load(f)

c_idx, l_idx = args.cam, args.lane
current_line = "line1"
points = []
mouse_pos = None

# initial capture
cam = cfg['cameras'][c_idx]
base_frame = capture_single_frame(cam['source'])
if base_frame is None:
    base_frame = np.zeros((480,640,3), dtype=np.uint8)

def mouse_callback(event, x, y, flags, param):
    global points, mouse_pos
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

win = "Lane Calibration"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 640, 480)
cv2.setMouseCallback(win, mouse_callback)

while True:
    cam = cfg['cameras'][c_idx]
    lane = cam['lanes'][l_idx]

    frame = base_frame.copy()

    # draw existing lines
    for ln in cam['lanes']:
        if 'line1' in ln:
            cv2.line(frame, tuple(ln['line1'][0]), tuple(ln['line1'][1]), (0,255,255), 3)
        if 'line2' in ln:
            cv2.line(frame, tuple(ln['line2'][0]), tuple(ln['line2'][1]), (0,0,255), 3)

    # preview line
    if len(points) == 1 and mouse_pos:
        color = (0,255,0) if current_line=="line1" else (255,0,0)
        cv2.line(frame, tuple(points[0]), mouse_pos, color, 2)

    # draw points
    for i, p in enumerate(points):
        cv2.circle(frame, tuple(p), 6, (0,255,0), -1)
        draw_text(frame, str(i+1), (p[0]+10, p[1]+10), 0.7)

    # commit line
    if len(points) == 2:
        lane[current_line] = [points[0], points[1]]
        print(f"[SET] Cam {c_idx} Lane {l_idx} {current_line}")
        points.clear()

    draw_mini_status(frame, cam, lane, current_line, points)

    cv2.imshow(win, frame)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('1'):
        current_line = "line1"; points.clear()
    elif key == ord('2'):
        current_line = "line2"; points.clear()
    elif key == ord('z') and points:
        points.pop()
    elif key == ord('r'):
        lane.pop(current_line, None); points.clear()
    elif key == ord('n'):
        l_idx = (l_idx + 1) % len(cam['lanes']); points.clear()
    elif key == ord('p'):
        l_idx = (l_idx - 1) % len(cam['lanes']); points.clear()
    elif key == ord('c'):
        c_idx = (c_idx + 1) % len(cfg['cameras'])
        l_idx = 0
        points.clear()
        base_frame = capture_single_frame(cfg['cameras'][c_idx]['source'])
        if base_frame is None:
            base_frame = np.zeros((720,1280,3), dtype=np.uint8)
    elif key == ord('s'):
        backup_settings()
        with open("config/settings.yaml", "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        print("[INFO] Settings saved")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
