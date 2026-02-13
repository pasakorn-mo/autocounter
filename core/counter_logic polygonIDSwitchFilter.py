import time
import numpy as np
import cv2
from datetime import datetime

class MultiLaneCounter:
    def __init__(self, lanes_config, distance_m=5.0, mode=3):
        self.lanes_config = lanes_config
        self.distance_m = distance_m
        
        # เก็บเวลาล่าสุดที่มีรถเข้าแต่ละเลน (สำหรับจัดการ ID switch)
        self.last_any_entry_time = {lane['id']: 0 for lane in lanes_config}
        
        self.lanes = {}
        for lane in lanes_config:
            l1 = np.array(lane['line1'], dtype=np.int32)
            l2 = np.array(lane['line2'], dtype=np.int32)
            
            poly_points = np.array([l1[0], l1[1], l2[1], l2[0]], dtype=np.int32)
            
            self.lanes[lane['id']] = {
                'poly': poly_points,
                'exit_line': tuple(lane['line2'])
            }

        self.tracked_objects = {} 
        self.counted_ids = {}
        self.last_reset_date = datetime.now().date()

    def _dist_point_to_segment(self, P, A, B):
        """คำนวณระยะห่างจากจุด P ไปยังเส้นตรง AB"""
        p = np.array(P)
        a = np.array(A)
        b = np.array(B)
        
        ab = b - a
        ap = p - a
        
        if np.linalg.norm(ab) == 0: 
            return np.linalg.norm(p - a)
        
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))
        
        projection = a + t * ab
        return np.linalg.norm(p - projection)

    def process(self, results):
        now = time.time()
        
        # Reset วันใหม่
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.counted_ids = {}
            self.tracked_objects = {}
            self.last_any_entry_time = {lid: 0 for lid in self.lanes.keys()}
            self.last_reset_date = today

        visual_events = []
        count_events = []

        if len(results) > 0 and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            current_frame_ids = set(ids)

            for i, obj_id in enumerate(ids):
                # 1. Cooldown Check: ป้องกันนับซ้ำ
                if obj_id in self.counted_ids and (now - self.counted_ids[obj_id]) < 2.0:
                    continue

                x1, y1, x2, y2 = boxes[i]
                curr_point = (int((x1 + x2) / 2), int(y2))

                # --- ตรวจสอบว่าอยู่ใน Lane ไหน ---
                inside_lane_id = None
                
                for l_id, lane_data in self.lanes.items():
                    is_inside = cv2.pointPolygonTest(lane_data['poly'], curr_point, False) >= 0

                    if is_inside:
                        inside_lane_id = l_id
                        
                        # CASE: เข้าโซนครั้งแรก
                        if obj_id not in self.tracked_objects:
                            self.tracked_objects[obj_id] = {
                                'entered_time': now,
                                'lane': l_id,
                                'last_seen': now
                            }
                            # บันทึกเวลาที่มีรถเข้าเลนนี้ล่าสุด
                            self.last_any_entry_time[l_id] = now
                            visual_events.append((l_id, 1))
                        else:
                            # CASE: ยังอยู่ในโซน
                            if self.tracked_objects[obj_id]['lane'] == l_id:
                                self.tracked_objects[obj_id]['last_seen'] = now
                        break  # เจอแล้วไม่ต้องตรวจเลนอื่น

                # --- Exit Logic ---
                if obj_id in self.tracked_objects and not inside_lane_id:
                    track_info = self.tracked_objects[obj_id]
                    lane_id = track_info['lane']
                    
                    exit_line = self.lanes[lane_id]['exit_line']
                    dist_to_exit = self._dist_point_to_segment(
                        curr_point, exit_line[0], exit_line[1]
                    )

                    # ตรวจสอบว่าออกทางที่ถูกต้อง (ใกล้ exit line)
                    if dist_to_exit < 150: 
                        # --- จัดการ ID Switch ---
                        actual_start_time = track_info['entered_time']
                        duration_by_id = now - actual_start_time
                        
                        # ถ้า duration สั้นเกินไป (< 0.5 วิ) = น่าจะเป็น ID switch
                        # ให้ใช้เวลาของรถคันก่อนหน้าแทน
                        if duration_by_id < 0.5 and self.last_any_entry_time[lane_id] > 0:
                            actual_start_time = self.last_any_entry_time[lane_id]
                        
                        duration = now - actual_start_time
                        
                        # คำนวณความเร็ว (ป้องกันหารด้วย 0)
                        speed_kmh = (self.distance_m / max(0.1, duration)) * 3.6
                        
                        # Filter ความเร็วที่สมเหตุสมผล
                        if 1 < speed_kmh < 220:
                            count_events.append((lane_id, confs[i], speed_kmh, obj_id))
                            self.counted_ids[obj_id] = now
                            visual_events.append((lane_id, 2))

                    # ลบออกจาก tracking
                    del self.tracked_objects[obj_id]

            # Cleanup: ลบ ID ที่หายไปนาน (2 วิ - กลางๆระหว่าง 1-3 วิ)
            stale_ids = [
                oid for oid, data in self.tracked_objects.items()
                if oid not in current_frame_ids and (now - data['last_seen']) > 2.0
            ]
            
            for oid in stale_ids:
                del self.tracked_objects[oid]
                
            # Cleanup Counted IDs (เคลียร์ ID ที่นับไปแล้วนาน)
            if len(self.counted_ids) > 200:
                self.counted_ids = {
                    k: v for k, v in self.counted_ids.items() 
                    if (now - v) < 60.0
                }

        return visual_events, count_events