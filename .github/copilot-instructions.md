# AutoCounter AI Agent Instructions

AutoCounter is a **multi-camera vehicle counting system** using YOLOv8 tracking and multiprocessing for real-time traffic monitoring.

## Architecture Overview

**Producer-Consumer Pattern**: YOLO-based `CameraAgent` processes per camera (spawned as separate processes), each pushes counting results to a shared `result_queue`. A single `ResultWorker` consumes from the queue and writes to SQLite database.

```
[RTSP Camera] → [CameraAgent (YOLO+Tracker)] → [result_queue] → [ResultWorker] → [SQLite DB]
                                              ↓
                                         [Streamlit Dashboard]
```

**Key Files**:
- [main.py](main.py) - Supervisor process; spawns agents + database worker, manages graceful shutdown
- [core/agent_worker.py](core/agent_worker.py) - Per-camera YOLO inference + lane counting logic
- [core/counter_logic.py](core/counter_logic.py) - MultiLaneCounter: polygon-based lane detection, speed calculation
- [core/result_worker.py](core/result_worker.py) - Consumer: batches transactions, writes images + metadata to DB
- [data/database.py](data/database.py) - SQLite schema: `transactions` (lane, speed, timestamp, image), `camera_status`
- [gui_dashboard.py](gui_dashboard.py) - Streamlit UI for live status + daily traffic analytics by lane/plaza

## Configuration & Conventions

**Settings YAML** ([config/settings.yaml](config/settings.yaml)):
- `cameras[]` - Array of camera configs with `id`, `plaza`, `source` (RTSP URL), and `lanes[]` (polygon + exit_line)
- `model.path` - Path to YOLOv8 checkpoint (e.g., `models/best26_trained_*.pt`)
- `processing.dist_between_lines_m` - Distance (meters) between lane entry/exit lines for speed estimation
- Model loading happens in agent: `YOLO(model_cfg['path'])`

**Lane Geometry**:
- Each lane has `line1` (entry) and `line2` (exit) as 2D coordinate pairs
- Counter creates **quad polygon** from these points for `cv2.pointPolygonTest()` to detect object entry
- Speed calculated: `speed_kmh = (distance_m / duration_s) * 3.6`

## Developer Workflows

**Running the system**:
```bash
python main.py                    # Starts supervisor + N agents + DB worker
```

**Running dashboard** (separate terminal):
```bash
streamlit run gui_dashboard.py    # Connects to same SQLite DB; auto-refresh every 10s
```

**Adding a new camera**:
1. Append entry to `cameras[]` in [config/settings.yaml](config/settings.yaml) with RTSP source and lane polygon coordinates
2. Ensure `models/` contains the selected model checkpoint
3. Restart `python main.py` - new CameraAgent spawns automatically

**Debugging individual cameras**:
- Logs stream to `logs/system.log` (rotating, 10MB each)
- Each agent logs as `Agent-{cam_id}` with frame stats every ~10 frames (frames, dropped, infer time, etc.)
- To inspect a single agent in isolation, create a minimal script that instantiates `CameraAgent` directly without the supervisor

## Key Patterns & Implementation Details

**Multiprocessing Spawn Context**: 
- [main.py#L33](main.py#L33-L35) forces `mp.set_start_method("spawn")` for PyTorch/OpenCV compatibility

**Lane Counting Logic**:
- [core/counter_logic.py](core/counter_logic.py): Objects tracked by `boxes.id` (YOLOv8 native tracker); counted only on exit
- **Cooldown (2s)**: Prevents double-counting same ID within 2 seconds
- **ID Switch Mitigation**: If an object exits too fast (<0.5s), use previous object's entry time to estimate real duration
- **Exit Detection**: Object must cross `line2` AND be within 150 pixels of exit line to trigger count event

**Image Capture Strategy**:
- Controlled via [core/result_worker.py](core/result_worker.py): `db.log_tx()` accepts frame as either **JPEG bytes** (preferred, low CPU) or **NumPy array**
- Images saved to `snapshots/{plaza}_{lane}_{timestamp}.jpg`
- Edge cases logged to `snapshots/edge_cases/` (threaded async save in [core/utils.py](core/utils.py))

**Database Pragmas**:
- [data/database.py#L19](data/database.py#L19-L21): WAL mode + NORMAL synchronous for concurrent read/write without long locks
- Tables indexed by `timestamp`, `gate`, `lane` for fast daily aggregation queries

**Queue Management**:
- `result_queue` sized 2000 items ([main.py#L52](main.py#L52)); agents drop frames if full (logged as "drops" stat)
- ResultWorker timeout=0.5s per item; respects `exit_event` for clean shutdown

## Integration Points & Dependencies

- **YOLOv8** (ultralytics): Model inference + built-in tracker; loaded per-agent
- **OpenCV** (cv2): Video capture, frame processing, geometry tests, image I/O
- **SQLite3**: Direct API (no ORM); concurrent writes via WAL mode
- **Streamlit**: Read-only dashboard against DB; fragments for 10s auto-refresh
- **PyYAML**: Configuration parsing

## Common Pitfalls & Conventions

1. **Process Spawning**: Always use `mp.get_context("spawn")` or set globally in main
2. **Logging in Processes**: Each process reconfigures logging (not inherited from parent); use `logging.getLogger(f"Agent-{id}")`
3. **Frame Timestamp**: Use `datetime.now()` in agent; sync to DB via `result_queue` items
4. **Lane Coordinates**: Define as `[[x1, y1], [x2, y2]]` pairs (pixel coords from frame intrinsics)
5. **Speed Filtering**: Range 1–220 km/h; outside range = discard (likely tracking error or occlusion)
6. **Shutdown Sequence**: Set `stop_event`, join agents (timeout 2s), terminate stragglers, stop DB worker last

## Testing & Validation

- No automated test suite; validation is live streaming (RTSP source required)
- To test lane detection offline: extract frame, visualize polygon with `cv2.drawContours()`, confirm detection zone
- To validate speed: known object crossing distance & time; compare reported speed to ground truth
