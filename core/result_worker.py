import multiprocessing as mp
import logging
import traceback
import queue
import time
from data.database import AutoCounterDB

class ResultWorker(mp.Process):
    def __init__(self, result_queue):
        super().__init__()
        self.result_queue = result_queue
        self.exit_event = mp.Event()

    def stop(self):
        self.exit_event.set()
        try:
            self.result_queue.put_nowait(("__STOP__",))
        except queue.Full:
            pass

    def run(self):
        # Setup Logging inside Process
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [DB_WORKER] %(message)s')
        db = AutoCounterDB()
        logging.info("✅ ResultWorker DB Connected.")

        stats_last = time.time()
        processed = 0

        while True:
            try:
                # ใช้ timeout เพื่อให้ Loop วนมาเช็ค exit_event ได้
                timeout = 0.5 if self.exit_event.is_set() else 1
                item = self.result_queue.get(timeout=timeout)
                
                cmd = item[0]
                if cmd == "__STOP__":
                    break
                if cmd == "TX":
                    # Unpack: Header, Plaza, Lane, Image, Speed
                    _, plaza, lane, frame, speed = item
                    db.log_tx(plaza, lane, frame, speed)
                    processed += 1
                    
                elif cmd == "STATUS":
                    # Unpack: Header, CamID, StatusStr
                    _, cam_id, status = item
                    db.update_status(cam_id, status)
                    processed += 1

                # Debug throughput every 10s
                now = time.time()
                if stats_last is None:
                    stats_last = now
                if now - stats_last >= 10:
                    try:
                        qsize = self.result_queue.qsize()
                    except Exception:
                        qsize = -1
                    logging.info("DBG DBWorker | items=%d in 10s | q=%s", processed, qsize)
                    processed = 0
                    stats_last = now
                    
            except queue.Empty:
                if self.exit_event.is_set():
                    break
                continue
            except Exception as e:
                logging.error(f"❌ Worker Error: {e}")
                traceback.print_exc()
