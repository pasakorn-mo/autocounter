import cv2
import time
import logging
import multiprocessing as mp
import numpy as np
import queue
from datetime import datetime
from ultralytics import YOLO
from core.counter_logic import MultiLaneCounter
from core.logging_config import setup_process_logging

class CameraAgent(mp.Process):
    def __init__(self, cam_config, model_config, proc_config, result_queue, stop_event):
        super().__init__()
        self.cfg = cam_config
        self.model_cfg = model_config
        self.proc_cfg = proc_config
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.cam_id = cam_config['id']

    def apply_lighting_filters(self, frame):
        try:
            gamma = 1.3 
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            frame = cv2.LUT(frame, table)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except:
            return frame

    def run(self):
        # Config Logging with centralized setup
        logger = setup_process_logging(f"Agent-{self.cam_id}", log_dir="logs", log_level=logging.INFO)

        try:
            model = YOLO(self.model_cfg['path'])
        except Exception as e:
            logger.error(f"❌ Load Model Failed: {e}")
            return

        distance_m = self.proc_cfg.get('dist_between_lines_m', 5.0)
        counter = MultiLaneCounter(self.cfg['lanes'], distance_m=distance_m)
        
        flash_status = {} 
        last_speeds = {} 
        last_heartbeat = 0
        
        cap = cv2.VideoCapture(self.cfg['source'])
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        logger.info(f"✅ Agent {self.cam_id} Started processing...")

        stats = {
            "frames": 0,
            "last_log": time.time(),
            "t_read": 0.0,
            "t_infer": 0.0,
            "t_logic": 0.0,
            "t_draw": 0.0,
            "drops": 0,
            "new_ids": 0,
            "lost_ids": 0,
        }
        prev_ids = set()

        def safe_put(item, timeout=0.2):
            try:
                self.result_queue.put(item, timeout=timeout)
                return True
            except queue.Full:
                stats["drops"] += 1
                logger.warning("⚠️ Result queue full. Dropping item.")
                return False

        frame_skip = int(self.proc_cfg.get("frame_skip", 0))

        while not self.stop_event.is_set():
            now = time.time()
            if now - last_heartbeat > 5:
                status = "Online" if cap.isOpened() else "Offline"
                safe_put(("STATUS", self.cam_id, status))
                last_heartbeat = now

            # Drop buffered frames to reduce RTSP latency
            for _ in range(frame_skip):
                cap.read()

            ret, frame = cap.read()
            if not ret:
                logger.warning(f"⚠️ Cam {self.cam_id} No signal. Reconnecting...")
                safe_put(("STATUS", self.cam_id, "Offline"))
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(self.cfg['source'])
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                continue

            t0 = time.time()
            #processed_frame = self.apply_lighting_filters(frame)
            processed_frame = frame
            t1 = time.time()
            tracker_cfg = self.model_cfg.get('tracker')
            if tracker_cfg:
                results = model.track(
                    processed_frame,
                    persist=True,
                    verbose=False,
                    conf=self.model_cfg.get('conf', 0.4),
                    tracker=tracker_cfg,
                )
            else:
                results = model.track(
                    processed_frame,
                    persist=True,
                    verbose=False,
                    conf=self.model_cfg.get('conf', 0.4),
                )
            t2 = time.time()
            visual_events, count_events = counter.process(results)
            t3 = time.time()

            # ID churn metrics
            curr_ids = set()
            if len(results) > 0 and results[0].boxes.id is not None:
                try:
                    curr_ids = set(results[0].boxes.id.cpu().numpy().astype(int).tolist())
                except Exception:
                    curr_ids = set()
            stats["new_ids"] += len(curr_ids - prev_ids)
            stats["lost_ids"] += len(prev_ids - curr_ids)
            prev_ids = curr_ids

            for l_id, line_num in visual_events:
                flash_status[(l_id, line_num)] = now + 0.5

            for l_id, conf, speed, obj_id in count_events:

                tx_data = ["TX", self.cfg['plaza'], l_id, None, speed]
                # ใน CameraAgent
                if self.proc_cfg.get('send_images', False):
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    payload_frame = buffer.tobytes() if ret else None
                else:
                    payload_frame = None

                safe_put(("TX", self.cfg['plaza'], l_id, payload_frame, speed))

                # --- [แก้ไข] ตัด Timestamp และ tag [INFO] ที่ทำเองออก ---
                # ให้ Logger จัดการ Time และ Level เอง
                log_msg = (
                    f"🚗 Plaza: {self.cfg['plaza']} | "
                    f"Lane: {l_id} | "
                    f"Speed: {speed:.0f} km/h | "
                    f"ID: {obj_id}"
                )
                
                logger.info(log_msg) 
                # ลบ print(log_msg) ออก เพราะ logger จะแสดงผลออก console ให้อยู่แล้วผ่าน StreamHandler
                
                last_speeds[l_id] = f"ID:{obj_id} {int(speed)} km/h"

            if not self.model_cfg.get('headless', False):
                display_frame = processed_frame.copy()
                
                if len(results) > 0 and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    for box, obj_id in zip(boxes, ids):
                        cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 165, 255), 2)
                        cv2.putText(display_frame, f"ID:{obj_id}", (box[0], box[1]-7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                for lane in self.cfg['lanes']:
                    l_id = lane['id']
                    c1 = (0, 255, 0) if now < flash_status.get((l_id, 1), 0) else (0, 0, 255)
                    c2 = (0, 255, 0) if now < flash_status.get((l_id, 2), 0) else (0, 0, 255)

                    cv2.line(display_frame, tuple(lane["line1"][0]), tuple(lane["line1"][1]), c1, 3)
                    cv2.line(display_frame, tuple(lane["line2"][0]), tuple(lane["line2"][1]), c2, 3)
                    
                    if l_id in last_speeds:
                        cv2.putText(display_frame, last_speeds[l_id], tuple(lane["line2"][1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow(f"Agent: {self.cam_id}", cv2.resize(display_frame, (640, 480)))
                if cv2.waitKey(1) == ord('q'):
                    self.stop_event.set()
                    break
                t4 = time.time()
            else:
                t4 = time.time()

            # --- Debug Stats (every 10s) ---
            stats["frames"] += 1
            stats["t_read"] += (t1 - t0)
            stats["t_infer"] += (t2 - t1)
            stats["t_logic"] += (t3 - t2)
            stats["t_draw"] += (t4 - t3)
            now = time.time()
            if now - stats["last_log"] >= 10:
                fps = stats["frames"] / (now - stats["last_log"])
                avg_read = (stats["t_read"] / max(1, stats["frames"])) * 1000
                avg_infer = (stats["t_infer"] / max(1, stats["frames"])) * 1000
                avg_logic = (stats["t_logic"] / max(1, stats["frames"])) * 1000
                avg_draw = (stats["t_draw"] / max(1, stats["frames"])) * 1000
                try:
                    qsize = self.result_queue.qsize()
                except Exception:
                    qsize = -1
                logger.info(
                    "DBG Agent %s | fps=%.1f read=%.1fms infer=%.1fms logic=%.1fms draw=%.1fms q=%s drops=%d new_id/s=%.2f lost_id/s=%.2f",
                    self.cam_id, fps, avg_read, avg_infer, avg_logic, avg_draw, qsize, stats["drops"],
                    stats["new_ids"] / max(1.0, (now - stats["last_log"])),
                    stats["lost_ids"] / max(1.0, (now - stats["last_log"]))
                )
                stats = {
                    "frames": 0,
                    "last_log": now,
                    "t_read": 0.0,
                    "t_infer": 0.0,
                    "t_logic": 0.0,
                    "t_draw": 0.0,
                    "drops": stats["drops"],
                    "new_ids": 0,
                    "lost_ids": 0,
                }

        cap.release()
        cv2.destroyAllWindows()
