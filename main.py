import time
import yaml
import logging
import multiprocessing as mp
import signal
import os
import sys
from logging.handlers import RotatingFileHandler
from core.agent_worker import CameraAgent
from core.result_worker import ResultWorker

def setup_root_logging(config):
    """Setup Central Logging"""
    log_dir = "logs"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    
    log_path = config.get('system', {}).get('log_file', 'logs/system.log')
    log_lvl = config.get('system', {}).get('log_level', 'INFO')
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # File Handler
    f_handler = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5)
    f_handler.setFormatter(formatter)
    
    # Console Handler
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(formatter)
    
    logging.basicConfig(level=getattr(logging, log_lvl), handlers=[c_handler, f_handler])

def main():
    # Force Spawn method for compatibility (PyTorch/OpenCV)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Load Config
    try:
        with open("config/settings.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Config Error: {e}")
        return

    setup_root_logging(cfg)
    logger = logging.getLogger("MainSupervisor")
    logger.info("==========================================")
    logger.info("=== 🚀 Starting AutoCounter System V5 ===")
    logger.info("==========================================")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=2000) # เพิ่ม Buffer เผื่อ Traffic สูง
    stop_event = ctx.Event()

    # 1. Start DB Worker (Consumer)
    db_worker = ResultWorker(result_queue)
    db_worker.start()
    logger.info("✅ DB Worker Started")

    # 2. Start Camera Agents (Producers)
    agents = []
    model_cfg = cfg['model']
    model_cfg['headless'] = cfg.get('system', {}).get('headless', False)

    for cam_cfg in cfg['cameras']:
        p = CameraAgent(cam_cfg, model_cfg, cfg['processing'], result_queue, stop_event)
        p.start()
        agents.append(p)
        logger.info(f"✅ Launched Agent: {cam_cfg['id']}")

    # 3. Graceful Shutdown Handler
    def signal_handler(sig, frame):
        logger.warning("\n🛑 Shutdown signal received! Terminating processes...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 4. Watchdog Loop
    try:
        while not stop_event.is_set():
            # ตรวจสอบว่า Process ตายหรือไม่ (สามารถเพิ่ม Auto-restart logic ได้ที่นี่)
            dead_agents = [p for p in agents if not p.is_alive()]
            if dead_agents:
                for p in dead_agents:
                    logger.error(f"⚠️ Agent Died: {p.name}. (Restart logic not implemented yet)")
                    agents.remove(p) # เอาออกจาก list เพื่อไม่ให้ log ซ้ำ
            
            if not db_worker.is_alive():
                logger.critical("❌ DB Worker Died! System Unstable.")
                stop_event.set()
                
            time.sleep(2)
            
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Cleaning up resources...")
        stop_event.set()
        
        # Stop Agents
        for p in agents:
            if p.is_alive():
                p.join(timeout=2)
                if p.is_alive(): p.terminate()
        
        # Stop DB Worker
        db_worker.stop()
        db_worker.join()
        
        logger.info("👋 System Shutdown Complete.")

if __name__ == "__main__":
    main()