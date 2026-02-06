from datetime import datetime
import time

class MultiLaneCounter:
    def __init__(self, lanes_config, distance_m=5.0):
        self.lanes = lanes_config
        self.distance_m = distance_m

        self.entry_times = {lane['id']: {} for lane in lanes_config}
        self.counted_ids = set()
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

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                ids = list(range(len(results[0].boxes)))
                
            confs = results[0].boxes.conf.cpu().numpy()

            current_frame_ids = set(ids)

            for i, obj_id in enumerate(ids):
                x1, y1, x2, y2 = boxes[i]

                cx = int((x1 + x2) / 2)
                cy = int(y2)  # bottom-center

                curr_point = (cx, cy)
                prev_point = self.previous_centroids.get(obj_id, curr_point)
                now = time.time()

                for lane in self.lanes:
                    l_id = lane['id']
                    l1s, l1e = map(tuple, lane['line1'])
                    l2s, l2e = map(tuple, lane['line2'])

                    if self.intersect(prev_point, curr_point, l1s, l1e):
                        if obj_id not in self.entry_times[l_id] and obj_id not in self.counted_ids:
                            self.entry_times[l_id][obj_id] = now

                    if self.intersect(prev_point, curr_point, l2s, l2e):
                        if obj_id in self.entry_times[l_id]:
                            duration = now - self.entry_times[l_id][obj_id]
                            if duration > 0.05:
                                speed_kmh = (self.distance_m / duration) * 3.6
                                if 0 < speed_kmh < 220:
                                    count_events.append(
                                        (l_id, confs[i], speed_kmh, obj_id)
                                    )
                                    self.counted_ids.add(obj_id)
                                    del self.entry_times[l_id][obj_id]

                self.previous_centroids[obj_id] = curr_point

            for k in list(self.previous_centroids.keys()):
                if k not in current_frame_ids:
                    del self.previous_centroids[k]

        return visual_events, count_events
