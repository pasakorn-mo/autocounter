import sqlite3
import time

DB_PATH = "data/autocounter.db"


def init_db(conn):
    # -------- vehicle events --------
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

    # -------- camera health --------
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

    while True:
        event = result_queue.get()
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

            conn.commit()

        except Exception as e:
            print("[ResultWorker ERROR]", e)
            time.sleep(0.2)

    conn.close()
    print("[ResultWorker] stopped")
