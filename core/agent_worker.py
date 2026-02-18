import cv2
import time
import multiprocessing as mp
import numpy as np
import queue

from ultralytics import YOLO

from core.counter_logic import MultiLaneCounter
from core.logging_config import setup_process_logging
from core.utils import save_edge_case_async


class CameraAgent(mp.Process):
    def __init__(self, cam_config, model_config, proc_config,
                 result_queue, stop_event, log_queue=None):
        super().__init__()
        self.cfg = cam_config
        self.model_cfg = model_config
        self.proc_cfg = proc_config
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.log_queue = log_queue
        self.cam_id = cam_config['id']

    def run(self):
        logger = setup_process_logging(
            f"Agent-{self.cam_id}",
            log_queue=self.log_queue
        )

        model_path = self.model_cfg['path']
        tracker_cfg = self.model_cfg.get("tracker")

        # ---------------------------
        # Load Model
        # ---------------------------
        try:
            logger.info(f"Loading model: {model_path}")
            model = YOLO(model_path, task="detect")
            logger.info("✅ Model loaded")
        except Exception as e:
            logger.error(f"❌ Load Model Failed: {e}")
            return

        # ---------------------------
        # Counter
        # ---------------------------
        distance_m = self.proc_cfg.get('dist_between_lines_m', 5.0)
        counter = MultiLaneCounter(self.cfg['lanes'], distance_m=distance_m)

        flash_status = {}
        last_speeds = {}
        last_heartbeat = 0

        cap = cv2.VideoCapture(self.cfg['source'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        logger.info(f"✅ Agent {self.cam_id} Started")

        # ---------------------------
        # Stats
        # ---------------------------
        stats = {
            "frames": 0,
            "last_log": time.time(),
            "t_infer": 0.0,
            "t_logic": 0.0,
        }

        # ---------------------------
        # Safe Queue
        # ---------------------------
        def safe_put(item, timeout=0.2):
            try:
                self.result_queue.put(item, timeout=timeout)
                return True
            except queue.Full:
                logger.warning("⚠️ Result queue full. Dropping item.")
                return False

        frame_skip = int(self.proc_cfg.get("frame_skip", 0))

        # ---------------------------
        # Main Loop
        # ---------------------------
        while not self.stop_event.is_set():

            now = time.time()

            # Heartbeat
            if now - last_heartbeat > 5:
                status = "Online" if cap.isOpened() else "Offline"
                safe_put(("STATUS", self.cam_id, status))
                last_heartbeat = now

            for _ in range(frame_skip):
                cap.read()

            ret, frame = cap.read()

            if not ret:
                logger.warning("⚠️ Camera lost. Reconnecting...")
                safe_put(("STATUS", self.cam_id, "Offline"))
                cap.release()
                time.sleep(3)
                cap = cv2.VideoCapture(self.cfg['source'])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            processed_frame = frame

            # ---------------------------
            # Detect + Track
            # ---------------------------
            t0 = time.time()

            results = model.track(
                processed_frame,
                persist=True,
                verbose=False,
                conf=self.model_cfg.get('conf', 0.3),
                classes=self.model_cfg.get('classes'),
                tracker=tracker_cfg
            )

            t1 = time.time()

            # ---------------------------
            # Edge Case Save
            # ---------------------------
            if self.proc_cfg.get('save_edge_cases', False):
                edge_conf_range = self.proc_cfg.get('edge_conf_range', [0.2, 0.4])
                if len(results) > 0 and hasattr(results[0], 'boxes'):
                    confs = results[0].boxes.conf.cpu().numpy()
                    for i, conf in enumerate(confs):
                        if edge_conf_range[0] <= conf <= edge_conf_range[1]:
                            cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                            save_edge_case_async(processed_frame, self.cam_id, conf, cls_id)

            # ---------------------------
            # Counter
            # ---------------------------
            visual_events, count_events = counter.process(results)

            t2 = time.time()

            # ---------------------------
            # Flash Trigger
            # ---------------------------
            for l_id, line_num in visual_events:
                flash_status[(l_id, line_num)] = now + 0.5

            # ---------------------------
            # TX
            # ---------------------------
            for l_id, conf, speed, obj_id in count_events:

                if self.proc_cfg.get('send_images', False):
                    ret, buffer = cv2.imencode('.jpg', frame)
                    payload_frame = buffer.tobytes() if ret else None
                else:
                    payload_frame = None

                safe_put(("TX",
                          self.cfg['plaza'],
                          l_id,
                          payload_frame,
                          speed))

                logger.info(
                    f"🚗 Plaza: {self.cfg['plaza']} | "
                    f"Lane: {l_id} | "
                    f"Speed: {speed:.0f} km/h | "
                    f"ID: {obj_id}"
                )

                last_speeds[l_id] = f"ID:{obj_id} {int(speed)} km/h"

            # ---------------------------
            # Display
            # ---------------------------
            if not self.model_cfg.get('headless', False):

                display = processed_frame.copy()

                # Draw boxes
                if len(results) > 0 and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)

                    for box, obj_id in zip(boxes, ids):
                        cv2.rectangle(display,
                                      (box[0], box[1]),
                                      (box[2], box[3]),
                                      (0, 165, 255), 2)
                        cv2.putText(display,
                                    f"ID:{obj_id}",
                                    (box[0], box[1] - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 165, 255),
                                    2)

                # Draw Lanes
                for lane in self.cfg['lanes']:
                    l_id = lane['id']

                    c1 = (0, 255, 0) if now < flash_status.get((l_id, 1), 0) else (0, 0, 255)
                    c2 = (0, 255, 0) if now < flash_status.get((l_id, 2), 0) else (0, 0, 255)

                    cv2.line(display,
                             tuple(lane["line1"][0]),
                             tuple(lane["line1"][1]),
                             c1, 3)

                    cv2.line(display,
                             tuple(lane["line2"][0]),
                             tuple(lane["line2"][1]),
                             c2, 3)

                    if l_id in last_speeds:
                        cv2.putText(display,
                                    last_speeds[l_id],
                                    tuple(lane["line2"][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 255),
                                    2)

                cv2.imshow(f"Agent: {self.cam_id}",
                           cv2.resize(display, (640, 480)))

                if cv2.waitKey(1) == ord('q'):
                    self.stop_event.set()
                    break

            # ---------------------------
            # Stats
            # ---------------------------
            stats["frames"] += 1
            stats["t_infer"] += (t1 - t0)
            stats["t_logic"] += (t2 - t1)

            if time.time() - stats["last_log"] >= 10:
                fps = stats["frames"] / 10
                logger.info(
                    f"DBG {self.cam_id} | FPS:{fps:.1f} "
                    f"Infer:{(stats['t_infer']/max(1,stats['frames'])*1000):.1f}ms "
                    f"Logic:{(stats['t_logic']/max(1,stats['frames'])*1000):.1f}ms"
                )
                stats = {
                    "frames": 0,
                    "last_log": time.time(),
                    "t_infer": 0.0,
                    "t_logic": 0.0,
                }

        cap.release()
        cv2.destroyAllWindows()
