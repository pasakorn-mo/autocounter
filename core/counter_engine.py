import cv2
import time
from ultralytics import YOLO
from core.track_state import TrackState
from core.geometry import crossed_line
from core.utils import draw_lanes, draw_tracks

class CounterEngine:
    def __init__(self, cam_cfg, system_cfg, result_queue):
        self.cam_id = cam_cfg["cam_id"]
        self.source = cam_cfg["source"]
        self.lanes = cam_cfg.get("lanes", [])

        self.model = YOLO(system_cfg["model"]["path"])
        self.conf = system_cfg["model"]["conf"]
        self.iou = system_cfg["model"]["iou"]
        self.ttl = system_cfg["processing"]["track_ttl_sec"]

        self.preview = system_cfg.get("processing", {}).get("preview", True)
        self.tracks = {}
        self.q = result_queue

    def run(self):
        cap = cv2.VideoCapture(self.source)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] {self.cam_id} frame grab failed")
                time.sleep(0.5)
                cap.release()
                cap = cv2.VideoCapture(self.source)
                continue

            ts = time.time()

            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )

            self._update_tracks(results, ts)
            self._cleanup(ts)

            # ---------- PREVIEW ----------
            if self.preview:
                draw_lanes(frame, self.lanes)

                if results and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids   = results[0].boxes.id.cpu().numpy().astype(int)
                    draw_tracks(frame, boxes, ids, self.tracks)

                cv2.imshow(f"Preview {self.cam_id}", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        if self.preview:
            cv2.destroyAllWindows()



    def _update_tracks(self, results, ts):
        if not results or results[0].boxes.id is None:
            return

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids   = results[0].boxes.id.cpu().numpy().astype(int)

        for box, oid in zip(boxes, ids):
            cx = int((box[0] + box[2]) / 2)
            cy = int(box[3])  # bottom-center
            cur_pos = (cx, cy)

            st = self.tracks.get(oid)

            if not st:
                self.tracks[oid] = TrackState(
                    object_id=oid,
                    last_pos=cur_pos,
                    last_ts=ts
                )
                continue

            prev_pos = st.last_pos

            for lane in self.lanes:
                lid = lane["id"]

                # ---- LINE 1 ----
                if st.crossed_l1_ts is None:
                    if crossed_line(lane["line1"], prev_pos, cur_pos):
                        st.crossed_l1_ts = ts
                        st.lane_id = lid
                        print(f"[L1] cam={self.cam_id} lane={lid} obj={oid}")
                        break

                # ---- LINE 2 (ต้องหลัง L1) ----
                elif (
                    st.crossed_l2_ts is None
                    and st.lane_id == lid
                    and crossed_line(lane["line2"], prev_pos, cur_pos)
                ):
                    st.crossed_l2_ts = ts
                    st.finalized = True

                    print(
                        f"[FINAL] cam={self.cam_id} "
                        f"lane={lid} obj={oid}"
                    )
                    break

            st.last_pos = cur_pos
            st.last_ts = ts


    def _cleanup(self, now):
        self.tracks = {
            oid: st for oid, st in self.tracks.items()
            if (not st.finalized) and (now - st.last_ts < self.ttl)
        }
