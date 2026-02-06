# workers/result_worker.py
import sqlite3
import time

DB_PATH = "data/autocounter.db"


def init_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS vehicle_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_ts REAL NOT NULL,
        plaza TEXT NOT NULL,
        lane_id TEXT NOT NULL
    )
    """)

    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_events_plaza_lane_ts
    ON vehicle_events (plaza, lane_id, event_ts)
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS camera_health (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        cam_id TEXT NOT NULL,
        plaza TEXT NOT NULL,
        status TEXT NOT NULL,
        fps REAL,
        last_frame_ts REAL
    )
    """)

    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_health_cam_ts
    ON camera_health (cam_id, ts)
    """)

    conn.commit()


def result_worker(result_queue):
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )
    init_db(conn)

    print("[ResultWorker] SQLite ready")

    pending = 0
    last_commit = time.time()

    while True:
        event = result_queue.get()

        # ---------- SHUTDOWN ----------
        if event is None:
            break

        try:
            # ---------- VEHICLE EVENT ----------
            if "lane_id" in event:
                conn.execute(
                    """
                    INSERT INTO vehicle_events
                    (event_ts, plaza, lane_id)
                    VALUES (?, ?, ?)
                    """,
                    (
                        event["event_ts"],
                        event["plaza"],
                        event["lane_id"]
                    )
                )
                pending += 1

            # ---------- CAMERA HEALTH ----------
            elif "cam_id" in event:
                conn.execute(
                    """
                    INSERT INTO camera_health
                    (ts, cam_id, plaza, status, fps, last_frame_ts)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["ts"],
                        event["cam_id"],
                        event["plaza"],
                        event["status"],
                        event.get("fps"),
                        event.get("last_frame_ts")
                    )
                )
                pending += 1

            # ---------- COMMIT POLICY ----------
            now = time.time()
            if pending >= 20 or now - last_commit >= 2:
                conn.commit()
                pending = 0
                last_commit = now

        except Exception:
            # ไม่ print เพื่อลด block
            time.sleep(0.1)

    # ---------- FINAL COMMIT ----------
    try:
        conn.commit()
    except:
        pass

    conn.close()
    print("[ResultWorker] stopped")
