import yaml
import multiprocessing as mp
from core.counter_engine import CounterEngine
from workers.result_worker import ResultWorker

def main():
    mp.set_start_method("spawn", force=True)

    cfg = yaml.safe_load(open("config.yaml"))

    result_queue = mp.Queue(1000)

    # (optional) DB worker
    result_worker = ResultWorker(result_queue)
    result_worker.start()

    processes = []

    for cam_cfg in cfg["cameras"]:
        p = mp.Process(
            target=CounterEngine(cam_cfg, cfg, result_queue).run
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
