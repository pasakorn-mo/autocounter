import multiprocessing as mp
import logging
import traceback
from data.database import AutoCounterDB

class ResultWorker(mp.Process):
    def __init__(self, result_queue):
        super().__init__()
        self.result_queue = result_queue
        self.exit_event = mp.Event()

    def stop(self): self.exit_event.set()

    def run(self):
        # Setup Logging inside Process
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [DB_WORKER] %(message)s')
        db = AutoCounterDB()
        logging.info("✅ ResultWorker DB Connected.")

        while not self.exit_event.is_set() or not self.result_queue.empty():
            try:
                # ใช้ timeout เพื่อให้ Loop วนมาเช็ค exit_event ได้
                item = self.result_queue.get(timeout=1)
                
                cmd = item[0]
                if cmd == "TX":
                    # Unpack: Header, Plaza, Lane, Image, Speed
                    _, plaza, lane, frame, speed = item
                    db.log_tx(plaza, lane, frame, speed)
                    
                elif cmd == "STATUS":
                    # Unpack: Header, CamID, StatusStr
                    _, cam_id, status = item
                    db.update_status(cam_id, status)
                    
            except mp.queues.Empty:
                continue
            except Exception as e:
                logging.error(f"❌ Worker Error: {e}")
                traceback.print_exc()