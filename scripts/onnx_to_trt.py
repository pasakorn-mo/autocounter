#!/usr/bin/env python3
"""Convert an ONNX model to a TensorRT engine file.

Usage:
  python scripts/onnx_to_trt.py --onnx models/model.onnx --engine models/model.engine --fp16

This uses the TensorRT Python API. Install TensorRT + its Python wheel on your system.
"""
import argparse
import sys
import os

def build_engine(onnx_path: str, engine_path: str, fp16: bool = False, max_workspace: int = 1<<30):
    try:
        import tensorrt as trt
    except Exception as e:
        print("TensorRT is not installed or cannot be imported:", e)
        sys.exit(2)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    try:
        network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    except Exception:
        # Fallback for older/newer APIs
        network = builder.create_network()

    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace
    if fp16:
        try:
            config.set_flag(trt.BuilderFlag.FP16)
        except Exception:
            print("FP16 flag not available on this TensorRT version; continuing without FP16")

    print("Building TensorRT engine (this may take a while)...")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build the engine")

    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"✅ Engine written to: {engine_path}")


def main():
    p = argparse.ArgumentParser(description="ONNX -> TensorRT engine converter")
    p.add_argument('--onnx', required=True, help='Path to ONNX file')
    p.add_argument('--engine', required=True, help='Output engine file path')
    p.add_argument('--fp16', action='store_true', help='Enable FP16 flag')
    p.add_argument('--workspace', type=int, default=1<<30, help='Max workspace size in bytes')
    args = p.parse_args()

    onnx_path = args.onnx
    engine_path = args.engine

    if not os.path.exists(onnx_path):
        print(f"ONNX file not found: {onnx_path}")
        sys.exit(1)

    build_engine(onnx_path, engine_path, fp16=args.fp16, max_workspace=args.workspace)


if __name__ == '__main__':
    main()
