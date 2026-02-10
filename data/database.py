# data/database.py
import sqlite3
import os
import logging
from datetime import datetime

class AutoCounterDB:
    def __init__(self, db_path="autocounter.db"):
        self.db_path = db_path
        if not os.path.exists("snapshots"):
            os.makedirs("snapshots")
        self._init_db()

    def _get_connection(self):
        try:
            conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
            # เปิด WAL Mode เพื่อให้อ่าน/เขียนพร้อมกันได้เร็วขึ้น ไม่ Lock ไฟล์นาน
            conn.execute("PRAGMA journal_mode=WAL;") 
            conn.execute("PRAGMA synchronous=NORMAL;")
            return conn
        except sqlite3.OperationalError as e:
            logging.error(f"❌ Database connection failed: {e}")
            raise

    def _init_db(self):
        with self._get_connection() as conn:
            # Table: Transactions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gate TEXT,
                    lane TEXT,
                    timestamp DATETIME,
                    image_path TEXT,
                    speed REAL DEFAULT 0
                )
            """)
            # Table: Camera Status
            conn.execute("""
                CREATE TABLE IF NOT EXISTS camera_status (
                    cam_id TEXT PRIMARY KEY,
                    status TEXT,
                    last_seen DATETIME
                )
            """)
            
            # Indexes for Speed
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_gate_lane ON transactions(gate, lane)")

    # --- Write Functions ---
    def log_tx(self, gate, lane, frame, speed=0):
        try:
            now = datetime.now()
            timestamp_str = now.strftime('%Y%m%d_%H%M%S_%f')
            img_path = None # เริ่มต้นด้วย None (กรณีปิดภาพ)

            # 1. จัดการเรื่องการบันทึกภาพ
            if frame is not None:
                import os
                # สร้างโฟลเดอร์ถ้ายังไม่มี
                os.makedirs("snapshots", exist_ok=True)
                
                img_path = f"snapshots/{gate}_{lane}_{timestamp_str}.jpg"

                if isinstance(frame, bytes):
                    # --- กรณีส่งเป็น JPEG Bytes (แนะนำ) ---
                    # เขียนไฟล์ binary ตรงๆ ไม่ผ่าน OpenCV ลด CPU ได้มหาศาล
                    with open(img_path, "wb") as f:
                        f.write(frame)
                else:
                    # --- กรณีส่งเป็น NumPy Array (ภาพดิบ) ---
                    import cv2
                    # เช็คอีกทีว่าภาพไม่ว่าง (กันเหนียว)
                    if frame.size > 0:
                        cv2.imwrite(img_path, frame)
                    else:
                        img_path = None

            # 2. บันทึกลงฐานข้อมูล
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO transactions (gate, lane, timestamp, image_path, speed) 
                    VALUES (?, ?, ?, ?, ?)
                """, (gate, lane, now.isoformat(), img_path, speed))
                
        except Exception as e:
            # ตรงนี้จะดักจับทั้ง Error จาก OpenCV และ Database
            logging.error(f"⚠️ Save TX Error: {e}")

    def update_status(self, cam_id, status):
        """อัปเดตสถานะกล้อง (Heartbeat)"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO camera_status (cam_id, status, last_seen)
                    VALUES (?, ?, ?)
                    ON CONFLICT(cam_id) DO UPDATE SET 
                        status=excluded.status, 
                        last_seen=excluded.last_seen
                """, (cam_id, status, datetime.now().isoformat()))
        except Exception as e:
            logging.error(f"⚠️ Update Status Error: {e}")

    # --- Read Functions (For GUI) ---
    def get_realtime_counts(self):
        """คืนค่าจำนวนรถแยกตามด่านและเลน ของวันนี้"""
        today = datetime.now().strftime('%Y-%m-%d')
        query = """
            SELECT gate, lane, COUNT(*) as total 
            FROM transactions 
            WHERE timestamp >= ?
            GROUP BY gate, lane
            ORDER BY gate, lane
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(query, (today,))
                return cursor.fetchall() # [(gate, lane, count), ...]
        except Exception as e:
            logging.error(f"DB Read Error: {e}")
            return []

    def get_statuses(self):
        """ดึงสถานะล่าสุดของทุกกล้อง"""
        query = "SELECT cam_id, status, last_seen FROM camera_status ORDER BY cam_id"
        try:
            with self._get_connection() as conn:
                return conn.execute(query).fetchall()
        except Exception as e:
            logging.error(f"DB Status Error: {e}")
            return []

    def get_hourly_counts(self):
        """ดึงข้อมูลรายชั่วโมงของวันนี้"""
        today = datetime.now().strftime('%Y-%m-%d')
        # SQLite strftime('%H') ดึงชั่วโมงจาก ISO String
        query = """
            SELECT gate, lane, strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM transactions
            WHERE timestamp >= ?
            GROUP BY gate, lane, hour
            ORDER BY gate, lane, hour
        """
        try:
            with self._get_connection() as conn:
                return conn.execute(query, (today,)).fetchall()
        except Exception as e:
            logging.error(f"DB Hourly Error: {e}")
            return []
            
    def get_average_speed_today(self):
        today = datetime.now().strftime('%Y-%m-%d')
        try:
            with self._get_connection() as conn:
                res = conn.execute("SELECT AVG(speed) FROM transactions WHERE timestamp >= ?", (today,)).fetchone()
                return res[0] if res[0] is not None else 0
        except:
            return 0

    def get_hourly_counts_range(self, start_date, end_date, plaza=None):
        """ดึงข้อมูลรายชั่วโมงตามช่วงวันที่เลือก และกรองตาม Plaza (ถ้ามี)"""
        query = """
            SELECT gate, lane, strftime('%Y-%m-%d', timestamp) as date, 
                   strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM transactions
            WHERE date(timestamp) BETWEEN ? AND ?
        """
        params = [start_date, end_date]
        
        if plaza and plaza != "All Plazas":
            query += " AND gate = ?"
            params.append(plaza)
            
        query += " GROUP BY gate, lane, date, hour ORDER BY date DESC, gate, lane, hour"
        
        try:
            with self._get_connection() as conn:
                return conn.execute(query, params).fetchall()
        except Exception as e:
            logging.error(f"DB Range Error: {e}")
            return []

    def get_all_plazas(self):
        """ดึงรายชื่อ Plaza ทั้งหมดที่มีในระบบ"""
        try:
            with self._get_connection() as conn:
                res = conn.execute("SELECT DISTINCT gate FROM transactions").fetchall()
                return [r[0] for r in res]
        except:
            return []