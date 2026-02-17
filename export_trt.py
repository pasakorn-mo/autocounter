from ultralytics import YOLO

# โหลดโมเดลเดิม
model = YOLO("models/best26_trained_cp200img_cp150img_t00-10-193img_cpmtc201img.pt")

# Export เป็น TensorRT
# format='engine': ระบุ format
# half=True: ใช้ FP16 (เร็วกว่า FP32 มาก และเสียความแม่นยำน้อยมาก)
# device=0: ใช้ GPU ตัวที่ 0
# imgsz=640: ล็อกขนาด Input (สำคัญมากสำหรับ TensorRT เพื่อประสิทธิภาพสูงสุด)
# simplify=True: ช่วยลดความซับซ้อนของ ONNX graph ก่อนแปลง
model.export(format="engine", device=0, half=True, imgsz=640, simplify=True)