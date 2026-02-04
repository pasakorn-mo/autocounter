import time

class CameraHealth:
    def __init__(self, cam_id, plaza):
        self.cam_id = cam_id
        self.plaza = plaza
        self.last_frame_ts = None
        self.frames = 0
        self.last_report = time.time()

    def tick(self):
        now = time.time()
        self.frames += 1
        self.last_frame_ts = now

        if now - self.last_report >= 60:
            fps = self.frames / (now - self.last_report)
            self.frames = 0
            self.last_report = now
            return {
                "ts": now,
                "cam_id": self.cam_id,
                "plaza": self.plaza,
                "status": "OK",
                "fps": round(fps, 2),
                "last_frame_ts": self.last_frame_ts
            }
        return None
