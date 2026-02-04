import cv2
import time
import multiprocessing as mp
from ultralytics import YOLO


# -----------------------
# Config
# -----------------------
CAMERAS = [
    {
        "cam_id": "CAM_01",
        "source": "rtsp://pi:PiPassword@10.10.79.82:554/cam/realmonitor?channel=1&subtype=1" 
    },
    {
        "cam_id": "CAM_02",
        "source": "rtsp://pi:PiPassword@10.10.71.82:554/cam/realmonitor?channel=1&subtype=1"
    },
    {
        "cam_id": "CAM_03",
        "source": "rtsp://pi:PiPassword@10.10.78.82:554/cam/realmonitor?channel=1&subtype=1"
    },
    {
        "cam_id": "CAM_04",
        "source": "rtsp://counter:counter123@10.10.71.83:554/cam/realmonitor?channel=1&subtype=1"
    },
    {
        "cam_id": "CAM_05",
        "source": "rtsp://counter:counter123@10.10.72.82:554/cam/realmonitor?channel=1&subtype=1"
    },
    {
        "cam_id": "CAM_06",
        "source": "rtsp://counter:counter123@10.10.71.84:554/cam/realmonitor?channel=1&subtype=1"
    }

]

MODEL_PATH = "models/best26_trained_cp200img_cp150img_t00-10-193img.pt"
CONF = 0.4
IOU = 0.5
SHOW = True


# -----------------------
# Per-Camera Process
# -----------------------
def camera_process(cam_cfg):
    cam_id = cam_cfg["cam_id"]
    source = cam_cfg["source"]

    print(f"🚀 [{cam_id}] starting")

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ [{cam_id}] cannot open source")
        return

    win_name = f"Detect+Track [{cam_id}]"

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        t0 = time.time()

        results = model.track(
            frame,
            persist=True,       # 🔥 tracker state ต่อกล้อง
            conf=CONF,
            iou=IOU,
            verbose=False
        )

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, oid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"{cam_id} ID:{oid} {conf:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        fps = 1 / (time.time() - t0)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2
        )

        if SHOW:
            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"🛑 [{cam_id}] stopped")


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    procs = []

    for cam in CAMERAS:
        p = mp.Process(target=camera_process, args=(cam,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
