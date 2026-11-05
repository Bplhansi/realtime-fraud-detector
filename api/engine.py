"""Loads the scoring engine: the C++ engine through pybind11 if it was built, otherwise a
pure-Python engine that computes the same features and runs the same ONNX model.

Both have the same methods:
    score(step, type, amount, orig, dest) -> {"score", "decision", "features", "latency_ns"}
    remember_file(path) -> number of past transactions added to history
    model_version, review, block
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from ml.data import TYPES, read_tx_file
from ml.features import FeatureStore

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ENGINE_DIR = REPO / "engine" / "build" / "python"


def read_thresholds(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


class PythonEngine:
    """Same results as the C++ engine, slower. Used when the C++ module isn't built."""

    kind = "python"

    def __init__(self, model_path: Path, thresholds_path: Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), opts,
                                             providers=["CPUExecutionProvider"])
        t = read_thresholds(thresholds_path)
        self.review = float(t["review"])
        self.block = float(t["block"])
        self.model_version = t.get("model_version", "unknown")
        self._store = FeatureStore()

    @property
    def accounts(self) -> int:
        return len(self._store)

    def score(self, step: int, type: str, amount: float, orig: int, dest: int) -> dict:
        t0 = time.perf_counter_ns()
        row = self._store.process(step, TYPES.index(type), amount, orig, dest)
        x = np.asarray([row], dtype=np.float32)
        p = float(self._session.run(["probabilities"], {"features": x})[0][0, 1])
        decision = "block" if p >= self.block else "review" if p >= self.review else "approve"
        return {"score": p, "decision": decision, "features": row,
                "latency_ns": time.perf_counter_ns() - t0}

    def remember_file(self, path: str | Path) -> int:
        txs = read_tx_file(Path(path))
        for t in txs:
            self._store.update(int(t["step"]), float(t["amount"]), int(t["orig"]), int(t["dest"]))
        return len(txs)


def _import_cpp(engine_dir: Path):
    if not engine_dir.is_dir():
        return None
    if str(engine_dir) not in sys.path:
        sys.path.insert(0, str(engine_dir))
    if hasattr(os, "add_dll_directory"):  # Windows: let Python find onnxruntime.dll
        os.add_dll_directory(str(engine_dir))
    try:
        return importlib.import_module("fraud_engine")
    except ImportError:
        return None


def load_engine(model_path: Path, thresholds_path: Path, prefer: str = "auto",
                engine_dir: Path | None = None):
    """prefer: "auto" (C++ if available), "cpp" (must be C++) or "python"."""
    if prefer not in ("auto", "cpp", "python"):
        raise ValueError(f"unknown engine choice: {prefer}")
    if prefer != "python":
        mod = _import_cpp(Path(engine_dir or os.environ.get("FD_ENGINE_DIR", DEFAULT_ENGINE_DIR)))
        if mod is not None:
            return _CppEngine(mod.Engine(str(model_path), str(thresholds_path)))
        if prefer == "cpp":
            raise RuntimeError("C++ engine module not found; build engine/ first")
    return PythonEngine(model_path, thresholds_path)


class _CppEngine:
    """Thin wrapper so both engines report .kind the same way."""

    kind = "cpp"

    def __init__(self, eng) -> None:
        self._eng = eng
        self.model_version = eng.model_version
        self.review = eng.review
        self.block = eng.block

    @property
    def accounts(self) -> int:
        return self._eng.accounts

    def score(self, step: int, type: str, amount: float, orig: int, dest: int) -> dict:
        return self._eng.score(step=step, type=type, amount=amount, orig=orig, dest=dest)

    def remember_file(self, path: str | Path) -> int:
        return self._eng.remember_file(str(path))
