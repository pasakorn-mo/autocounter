import cv2
import time
from ultralytics import YOLO

from core.health_monitor import CameraHealth
from core.multi_lane_counter import MultiLaneCounter
from core.utils import draw_lanes

class CounterEngine:
    def __init__(self, cam_cfg, system_cfg, result_queue):
        self.cam_id = cam_cfg["cam_id"]
        self.source = cam_cfg["source"]
        self.plaza = cam_cfg.get("plaza", "")
        self.lanes = cam_cfg.get("lanes", [])

        self.model_path = system_cfg["model"]["path"]
        self.conf = system_cfg["model"]["conf"]
        self.iou = system_cfg["model"]["iou"]

        self.preview = system_cfg.get("processing", {}).get("preview", False)
        self.q = result_queue

        self.model = None
        self.counter = MultiLaneCounter(self.lanes)
        self.health_monitor = CameraHealth(self.cam_id, self.plaza)

    def run(self):
        self.model = YOLO(self.model_path)
        cap = cv2.VideoCapture(self.source)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] {self.cam_id} frame grab failed")
                time.sleep(0.5)
                cap.release()
                cap = cv2.VideoCapture(self.source)
                continue

            # ---------- health ----------
            health_event = self.health_monitor.tick()
            if health_event:
                self.q.put(health_event)

            # ==================================================
            # YOLO DETECT (ใช้ predict แทน track)
            # ==================================================
            results = self.model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )

            # DEBUG: ดูว่ามี detect จริงไหม
            if results and results[0].boxes is not None:
                print(f"[{self.cam_id}] DETECT:", len(results[0].boxes))

            # ==================================================
            # COUNT
            # ==================================================
            visual_events, count_events = self.counter.process(results)

            for lane_id, conf, speed, obj_id in count_events:
                event = {
                    "event_ts": time.time(),
                    "plaza": self.plaza,
                    "lane_id": lane_id,
                    "speed": round(speed, 2)
                }
                try:
                    self.q.put(event, timeout=0.01)
                except:
                    pass

            # ---------- PREVIEW ----------
            if self.preview:
                draw_lanes(frame, self.lanes)
                cv2.imshow(f"Preview {self.cam_id}", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        if self.preview:
            cv2.destroyAllWindows()

