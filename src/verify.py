#!/usr/bin/env python3
"""
verify.py — Верификация результатов nnc-codegen через сравнение с PyTorch/ONNX.

Запуск:
    pip install torch onnx onnxruntime numpy
    python3 scripts/verify.py

Что делает скрипт:
  1. Создаёт простую модель (Linear -> ReLU -> Linear) в PyTorch
  2. Экспортирует её в ONNX
  3. Запускает inference через onnxruntime (эталон)
  4. Генерирует MLIR через nnc-codegen (демо-режим)
  5. Показывает расхождение (пока nnc не поддерживает исполнение — шаг 5 служит
     отправной точкой для дальнейшей интеграции с MLIR execution engine)

TODO (расширение):
  • После добавления ExecutionEngine в nnc-codegen:
    - скомпилировать .mlir → .so (shared library)
    - вызвать через ctypes и сравнить с onnxruntime output
"""

import os
import sys
import subprocess
import numpy as np

def try_import(module_name):
    try:
        return __import__(module_name)
    except ImportError:
        print(f"[skip] {module_name} not installed — skipping that step")
        return None

torch = try_import("torch")
onnx  = try_import("onnx")
ort   = try_import("onnxruntime")

class SimpleMLP(object):
    """Имитация простого MLP без зависимостей от torch (fallback)."""
    pass

def run_pytorch_reference(input_data: np.ndarray):
    """Эталонный запуск через PyTorch."""
    if torch is None:
        print("[skip] PyTorch not available")
        return None

    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )
    model.eval()

    x = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)

    print("[PyTorch] Output shape:", out.shape)
    print("[PyTorch] Output:", out.numpy())
    return model, out.numpy()


def export_to_onnx(model, input_shape=(1, 16), path="model.onnx"):
    """Экспорт модели в ONNX."""
    if torch is None or onnx is None:
        return None

    import torch
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    print(f"[ONNX] Exported to {path}")
    return path


def run_onnxruntime(onnx_path: str, input_data: np.ndarray):
    """Запуск inference через ONNXRuntime."""
    if ort is None:
        print("[skip] onnxruntime not available")
        return None

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    out  = sess.run(None, {"input": input_data})[0]
    print("[ONNXRuntime] Output:", out)
    return out


def run_nnc_codegen_demo():
    """Запуск nnc-codegen в демо-режиме и проверка, что MLIR создаётся."""
    exe = os.path.join(os.path.dirname(__file__),
                       "..", "build", "nnc-codegen")
    if not os.path.exists(exe):
        # Пробуем найти в PATH
        exe = "nnc-codegen"

    cmd = [exe, "--verbose", "--no-passes",
           "--out", "/tmp/nnc_verify", "--outdir", "/tmp"]
    print(f"\n[nnc-codegen] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0:
            mlir_path = "/tmp/nnc_verify.mlir"
            if os.path.exists(mlir_path):
                print(f"[nnc-codegen] Generated MLIR ({os.path.getsize(mlir_path)} bytes):")
                with open(mlir_path) as f:
                    print(f.read())
            return True
        else:
            print(f"[nnc-codegen] stderr: {result.stderr}")
            return False
    except FileNotFoundError:
        print(f"[skip] nnc-codegen binary not found at '{exe}'.\n"
              "       Build the project first:  cd build && make")
        return False


def compare_outputs(ref: np.ndarray, ours: np.ndarray, tol: float = 1e-4):
    """Сравнение выходов с допуском."""
    if ref is None or ours is None:
        return
    diff = np.max(np.abs(ref - ours))
    status = "PASS" if diff < tol else "FAIL"
    print(f"\n[compare] Max abs diff = {diff:.6f}  [{status}]")


# ---- main ----
if __name__ == "__main__":
    np.random.seed(42)
    input_data = np.random.randn(1, 16).astype(np.float32)
    print("Input shape:", input_data.shape)

    result = run_pytorch_reference(input_data)
    if result:
        model, ref_out = result
        onnx_path = export_to_onnx(model, input_data.shape)
        if onnx_path:
            ort_out = run_onnxruntime(onnx_path, input_data)
            compare_outputs(ref_out, ort_out)

    run_nnc_codegen_demo()

    print("\nDone. For full verification:")
    print("  1. Extend Codegen with MLIR ExecutionEngine support")
    print("  2. Call the compiled function via ctypes")
    print("  3. Compare numerics with ref_out above")
