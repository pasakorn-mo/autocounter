import time
import numpy as np
import cv2
from datetime import datetime

class MultiLaneCounter:
    def __init__(self, lanes_config, distance_m=5.0, mode=3):
        # mode ไม่ได้ใช้ใน Logic Polygon แต่ใส่ไว้เพื่อให้ Interface เหมือนเดิม
        self.lanes_config = lanes_config
        self.distance_m = distance_m
        
        self.lanes = {}
        # แปลง Config เส้นคู่ ให้เป็น Polygon (สี่เหลี่ยม) ตั้งแต่เริ่ม
        for lane in lanes_config:
            # แปลงพิกัดเป็น Numpy Array
            l1 = np.array(lane['line1'], dtype=np.int32)
            l2 = np.array(lane['line2'], dtype=np.int32)
            
            # เรียงจุด: L1_start -> L1_end -> L2_end -> L2_start 
            # เพื่อสร้างเป็นพื้นที่ปิด (Closed Polygon) ไม่ให้เส้นไขว้กัน
            poly_points = np.array([
                l1[0], l1[1], l2[1], l2[0]
            ], dtype=np.int32)
            
            self.lanes[lane['id']] = {
                'poly': poly_points,       # พื้นที่ตรวจจับ
                'exit_line': tuple(lane['line2']) # เส้นออก (ใช้เช็คตอนจบ)
            }

        # เก็บสถานะรถที่กำลังอยู่ในโซน {obj_id: {'entered_time': float, 'lane': str}}
        self.tracked_objects = {} 
        
        # ป้องกันการนับซ้ำ {obj_id: timestamp}
        self.counted_ids = {}
        self.last_reset_date = datetime.now().date()

    def _dist_point_to_segment(self, P, A, B):
        """คำนวณระยะห่างจากจุด P ไปยังเส้นตรง AB (ใช้เช็คว่าออกถูกทางไหม)"""
        p = np.array(P)
        a = np.array(A)
        b = np.array(B)
        
        # เวกเตอร์ AB
        ab = b - a
        # เวกเตอร์ AP
        ap = p - a
        
        # หา Projection ของ P บนเส้น AB
        if np.linalg.norm(ab) == 0: return np.linalg.norm(p - a)
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0, min(1, t)) # Clamp ให้จุดอยู่บน Segment
        
        projection = a + t * ab
        return np.linalg.norm(p - projection)

    def process(self, results):
        now = time.time()
        
        # Reset วันใหม่ (ถ้าจำเป็น)
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.counted_ids = {}
            self.tracked_objects = {}
            self.last_reset_date = today

        visual_events = []
        count_events = []

        if len(results) > 0 and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            current_frame_ids = set(ids)

            for i, obj_id in enumerate(ids):
                # 1. Cooldown Check: ถ้าเพิ่งนับไปไม่ถึง 2 วินาที ให้ข้าม (กันนับเบิ้ล)
                if obj_id in self.counted_ids and (now - self.counted_ids[obj_id]) < 2.0:
                    continue

                x1, y1, x2, y2 = boxes[i]
                # ใช้ Bottom Center (เท้า) แม่นยำที่สุดสำหรับ Polygon บนพื้นถนน
                curr_point = (int((x1 + x2) / 2), int(y2))

                # --- ตรวจสอบว่าจุดนี้อยู่ใน Lane ไหนบ้าง ---
                inside_any_lane = False
                
                for l_id, lane_data in self.lanes.items():
                    # cv2.pointPolygonTest: คืนค่า >= 0 แปลว่า อยู่ใน หรือ บนเส้น
                    # (False คือ measureDist=False เอาแค่ True/False เร็วมาก)
                    is_inside = cv2.pointPolygonTest(lane_data['poly'], curr_point, False) >= 0

                    if is_inside:
                        inside_any_lane = True
                        
                        # CASE: เข้าโซนครั้งแรก (Entry)
                        if obj_id not in self.tracked_objects:
                            self.tracked_objects[obj_id] = {
                                'entered_time': now,
                                'lane': l_id,
                                'last_seen': now
                            }
                            visual_events.append((l_id, 1)) # ส่ง Event เส้นแรก (Entry)
                        else:
                            # CASE: ยังอยู่ในโซน (Update)
                            # ถ้าข้ามเลนในโซน ให้ยึดเลนแรกที่เข้า หรือเปลี่ยนตามใจชอบ (ที่นี่เลือกยึดอันเดิม)
                            if self.tracked_objects[obj_id]['lane'] == l_id:
                                self.tracked_objects[obj_id]['last_seen'] = now

                # --- ตรวจสอบการ "หลุด" ออกจากโซน (Exit Logic) ---
                # ถ้า ID นี้เคยถูก Track แต่เฟรมนี้ "ไม่อยู่ในเลนไหนเลย" -> แปลว่าเพิ่งออก
                if obj_id in self.tracked_objects and not inside_any_lane:
                    track_info = self.tracked_objects[obj_id]
                    lane_id = track_info['lane']
                    
                    # ตรวจสอบว่า ออกทางไหน? (ต้องออกใกล้ Line 2 เท่านั้น)
                    exit_line = self.lanes[lane_id]['exit_line']
                    dist_to_exit = self._dist_point_to_segment(
                        curr_point, exit_line[0], exit_line[1]
                    )

                    # Threshold: ระยะห่างจาก Line 2 ที่ยอมรับได้ (Pixel)
                    # ถ้าออกห่างจากเส้น 2 ไม่เกิน 150 px ถือว่าผ่านเส้น 2 สำเร็จ
                    # (ป้องกันกรณีย้อนศรออกทาง Line 1 หรือออกข้างๆ ไกลๆ)
                    if dist_to_exit < 150: 
                        duration = now - track_info['entered_time']
                        
                        # คำนวณความเร็ว
                        # max(0.05, ...) ป้องกันหารด้วย 0 หรือเวลาน้อยเกินไป
                        speed_kmh = (self.distance_m / max(0.05, duration)) * 3.6
                        
                        # Filter ความเร็วที่เพี้ยนมากๆ (เช่น 1000 km/h)
                        if 1 < speed_kmh < 250:
                            count_events.append((lane_id, confs[i], speed_kmh, obj_id))
                            self.counted_ids[obj_id] = now
                            visual_events.append((lane_id, 2)) # ส่ง Event เส้นสอง (Exit/Counted)

                    # ลบออกจาก tracking ทันทีที่หลุดโซน
                    del self.tracked_objects[obj_id]

            # Cleanup: ลบ ID ที่หายไปจากจอนานเกิน 3 วินาที (ป้องกัน Memory Leak)
            # กรณีรถจอดในโซนนานๆ แล้วหายวูบไป
            stale_ids = []
            for oid, data in self.tracked_objects.items():
                if oid not in current_frame_ids and (now - data['last_seen']) > 3.0:
                    stale_ids.append(oid)
            
            for oid in stale_ids:
                del self.tracked_objects[oid]
                
            # Cleanup Counted IDs (เคลียร์ทุก 1 นาที)
            if len(self.counted_ids) > 200:
                self.counted_ids = {k: v for k, v in self.counted_ids.items() if (now - v) < 60.0}

        return visual_events, count_events