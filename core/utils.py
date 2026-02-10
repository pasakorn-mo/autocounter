import cv2
import threading
import os
from datetime import datetime

def save_image_thread(frame, filename):
    try:
        cv2.imwrite(filename, frame)
    except Exception as e:
        print(f"Save Error: {e}")

def save_edge_case_async(frame, cam_id, conf, cls_id):
    """บันทึกภาพ Edge Case ลงโฟลเดอร์แยก"""
    folder = "snapshots/edge_cases"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # ตั้งชื่อไฟล์ระบุความมั่นใจ
    filename = f"{folder}/{cam_id}_{timestamp}_conf{int(conf*100)}_cls{int(cls_id)}.jpg"
    
    threading.Thread(
        target=save_image_thread, 
        args=(frame.copy(), filename), 
        daemon=True
    ).start()

def async_save_image(path, frame):
    """Save generic image async"""
    threading.Thread(
        target=save_image_thread, 
        args=(frame, path), 
        daemon=True
    ).start()