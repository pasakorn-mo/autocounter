# main.py
import yaml
import multiprocessing as mp
import signal
import sys
import time

from core.counter_engine import CounterEngine
from workers.result_worker import result_worker


def camera_process(cam_cfg, system_cfg, result_queue):
    """
    process ลูกสำหรับแต่ละกล้อง
    สร้าง CounterEngine ภายใน process (สำคัญกับ CUDA)
    """
    engine = CounterEngine(cam_cfg, system_cfg, result_queue)
    engine.run()


def main():
    mp.set_start_method("spawn", force=True)

    # ---------- LOAD CONFIG ----------
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    result_queue = mp.Queue(maxsize=1000)

    # ---------- RESULT WORKER ----------
    result_proc = mp.Process(
        target=result_worker,
        args=(result_queue,),
        daemon=False   # ❗ ห้าม daemon ไม่งั้นข้อมูลหาย
    )
    result_proc.start()

    processes = []

    # ---------- CAMERA PROCESSES ----------
    for cam_cfg in cfg["cameras"]:
        p = mp.Process(
            target=camera_process,
            args=(cam_cfg, cfg, result_queue),
            daemon=False
        )
        p.start()
        processes.append(p)

    # ---------- SIGNAL HANDLER ----------
    def shutdown(sig, frame):
        print("\n[MAIN] Shutting down...")

        # หยุด camera processes
        for p in processes:
            if p.is_alive():
                p.terminate()

        # ส่ง sentinel ให้ result_worker ปิดตัว
        result_queue.put(None)

        time.sleep(1)

        if result_proc.is_alive():
            result_proc.join(timeout=5)

        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ---------- WAIT ----------
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
