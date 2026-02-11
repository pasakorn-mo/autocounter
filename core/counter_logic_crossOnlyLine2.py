import time
from datetime import datetime

class MultiLaneCounter:
    def __init__(self, lanes_config, distance_m=5.0, mode=3):
        self.lanes = lanes_config
        self.distance_m = distance_m
        self.mode = 1 # เลือก 1, 2, หรือ 3
        
        # เก็บเวลาล่าสุดที่มี "วัตถุใดๆ" ทับเส้น 1 {lane_id: timestamp}
        self.last_any_entry = {lane['id']: 0 for lane in lanes_config}
        
        # เก็บเวลาเข้าแยกตาม ID {lane_id: {obj_id: timestamp}}
        self.entry_times = {lane['id']: {} for lane in lanes_config}
        
        # ป้องกันการนับ ID เดิมซ้ำในระยะเวลาสั้นๆ (Cooldown)
        self.counted_ids = {} # {obj_id: timestamp}
        self.last_reset_date = datetime.now().date()
        self.previous_centroids = {}

    def ccw(self, A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def intersect(self, A, B, C, D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

    def process(self, results):
        now = time.time()
        visual_events = []
        count_events = []

        if len(results) > 0 and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            current_frame_ids = set(ids)

            for i, obj_id in enumerate(ids):
                # ป้องกันนับซ้ำ: ถ้า ID นี้เพิ่งนับไปไม่ถึง 2 วินาที ให้ข้าม
                if obj_id in self.counted_ids and (now - self.counted_ids[obj_id]) < 2.0:
                    continue

                x1, y1, x2, y2 = boxes[i]
                curr_point = (int((x1 + x2) / 2), int(y2))
                prev_point = self.previous_centroids.get(obj_id, curr_point)

                for lane in self.lanes:
                    l_id = lane['id']
                    
                    # --- CHECK LINE 1 (ENTRY) ---
                    if self.intersect(prev_point, curr_point, tuple(lane['line1'][0]), tuple(lane['line1'][1])):
                        visual_events.append((l_id, 1))
                        self.last_any_entry[l_id] = now  # เก็บเวลา "ใครก็ได้" ที่เข้า
                        self.entry_times[l_id][obj_id] = now # เก็บเวลา "ID นี้" ที่เข้า

                    # --- CHECK LINE 2 (EXIT) ---
                    if self.intersect(prev_point, curr_point, tuple(lane['line2'][0]), tuple(lane['line2'][1])):
                        visual_events.append((l_id, 2))
                        
                        can_count = False
                        speed_kmh = 0
                        start_time = 0

                        # Logic แบบที่ 1: ตัดเส้น 2 ก็นับเลย
                        if self.mode == 1:
                            can_count = True

                        # Logic แบบที่ 2: ใครก็ได้ตัดเส้น 1 แล้วตามด้วยเส้น 2 ภายใน 3 วิ
                        elif self.mode == 2:
                            if (now - self.last_any_entry[l_id]) <= 3.0:
                                can_count = True
                                start_time = self.last_any_entry[l_id]

                        # Logic แบบที่ 3: ภายใน 3 วิใครก็ได้ / ถ้าเกิน 3 วิ ต้อง ID เดียวกัน
                        elif self.mode == 3:
                            # เช็คแบบ Fast Pass (ใครก็ได้ 3 วิ)
                            if (now - self.last_any_entry[l_id]) <= 3.0:
                                can_count = True
                                start_time = self.last_any_entry[l_id]
                            # เช็คแบบ ID Strict (ถ้าเกิน 3 วิ ต้อง ID เดิม)
                            elif obj_id in self.entry_times[l_id]:
                                can_count = True
                                start_time = self.entry_times[l_id][obj_id]

                        # บันทึกผล
                        if can_count:
                            if start_time > 0:
                                duration = now - start_time
                                speed_kmh = (self.distance_m / max(0.01, duration)) * 3.6
                            
                            count_events.append((l_id, confs[i], speed_kmh, obj_id))
                            self.counted_ids[obj_id] = now
                            # เคลียร์ข้อมูลหลังนับเสร็จ
                            self.last_any_entry[l_id] = 0 
                            if obj_id in self.entry_times[l_id]:
                                del self.entry_times[l_id][obj_id]

                self.previous_centroids[obj_id] = curr_point

            # Cleanup
            self.previous_centroids = {k: v for k, v in self.previous_centroids.items() if k in current_frame_ids}
            
            # ล้างประวัติ ID เก่าๆ ใน counted_ids ที่เกิน 1 นาที เพื่อไม่ให้ RAM บวม
            if len(self.counted_ids) > 100:
                self.counted_ids = {k: v for k, v in self.counted_ids.items() if (now - v) < 60.0}

        return visual_events, count_events