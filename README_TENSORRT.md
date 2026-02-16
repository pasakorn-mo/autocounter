# Running AutoCounter with TensorRT

This guide shows how to convert an existing ONNX model to a TensorRT engine and run AutoCounter using the engine.

Prerequisites
- CUDA + cuDNN installed
- TensorRT toolkit and Python wheel installed for your Python environment
- `pycuda` may be required if you plan to write custom inference (not required for the conversion script)

1) Convert ONNX → TensorRT engine

Run the included script:

```bash
python3 scripts/onnx_to_trt.py \
  --onnx models/best26_trained_cp200img_cp150img_t00-10-193img.onnx \
  --engine models/best26_trained_cp200img.engine \
  --fp16
```

Adjust `--workspace` if you need a different workspace size.

2) Point AutoCounter to the engine

- Edit `config/settings.yaml` and set `model.path` to the generated engine file (for example `models/best26_trained_cp200img.engine`).
- Optionally add a key `backend: tensorrt` under `model` so you (and future code) are aware you're using TRT.

Example (snippet):

```yaml
model:
  path: models/best26_trained_cp200img.engine
  conf: 0.4
  backend: tensorrt
```

3) Run AutoCounter

Run as usual:

```bash
python3 main.py
```

Notes
- The current `CameraAgent` uses the `ultralytics.YOLO` API. Some Ultralyics releases can load TensorRT engines directly if you provide an engine path; if your installed ultralytics build does not support direct TRT engines, you'll need to implement a small inference wrapper (replace model.track calls) that runs the TensorRT engine and provides detections in the same format the rest of the pipeline expects.
- If you want, I can add a TensorRT inference wrapper that replicates the shape of `ultralytics` results and integrates into `CameraAgent`.
