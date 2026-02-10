import time
from datetime import datetime

class MultiLaneCounter:
    def __init__(self, lanes_config, distance_m=5.0):
        self.lanes = lanes_config
        self.distance_m = distance_m
        
        # เก็บเวลาเข้า {lane_id: {obj_id: timestamp}}
        self.entry_times = {lane['id']: {} for lane in lanes_config}
        
        # เก็บ ID ที่นับไปแล้วเพื่อป้องกันการนับซ้ำ
        self.counted_ids = set()
        
        # เก็บพิกัดศูนย์กลางของเฟรมที่แล้ว {obj_id: (x, y)}
        self.previous_centroids = {}
        
        self.last_reset_date = datetime.now().date()

    def ccw(self, A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def intersect(self, A, B, C, D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

    def process(self, results):
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.entry_times = {lane['id']: {} for lane in self.lanes}
            self.counted_ids = set()
            self.previous_centroids = {}
            self.last_reset_date = today

        visual_events = []
        count_events = []

        if len(results) > 0 and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            current_frame_ids = set(ids)

            for i, obj_id in enumerate(ids):
                x1, y1, x2, y2 = boxes[i]
                
                # --- [จุดสำคัญที่แก้ไข] ใช้ Bottom Center (เท้า) แทน Center (เอว) ---
                cx = int((x1 + x2) / 2)
                cy = int(y2) 
                # -----------------------------------------------------------
                
                curr_point = (cx, cy)
                prev_point = self.previous_centroids.get(obj_id, curr_point)
                now = time.time()

                for lane in self.lanes:
                    l_id = lane['id']
                    line1_start = tuple(lane['line1'][0])
                    line1_end   = tuple(lane['line1'][1])
                    line2_start = tuple(lane['line2'][0])
                    line2_end   = tuple(lane['line2'][1])

                    # 1. เช็คเข้า
                    if self.intersect(prev_point, curr_point, line1_start, line1_end):
                        visual_events.append((l_id, 1))
                        if obj_id not in self.entry_times[l_id] and obj_id not in self.counted_ids:
                            self.entry_times[l_id][obj_id] = now

                    # 2. เช็คออก
                    if self.intersect(prev_point, curr_point, line2_start, line2_end):
                        visual_events.append((l_id, 2))
                        if obj_id in self.entry_times[l_id]:
                            start_time = self.entry_times[l_id][obj_id]
                            duration = now - start_time
                            
                            if duration > 0.05: # ลด Threshold เวลาลงนิดหน่อยเผื่อรถเร็วมาก
                                speed_kmh = (self.distance_m / duration) * 3.6
                                if 0 < speed_kmh < 220: 
                                    count_events.append((l_id, confs[i], speed_kmh, obj_id))
                                    self.counted_ids.add(obj_id)
                                    del self.entry_times[l_id][obj_id]

                self.previous_centroids[obj_id] = curr_point

            # Cleanup
            stale_ids = [k for k in self.previous_centroids.keys() if k not in current_frame_ids]
            for k in stale_ids:
                del self.previous_centroids[k]

        return visual_events, count_events