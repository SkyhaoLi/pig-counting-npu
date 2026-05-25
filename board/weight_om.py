"""ACL-based weight regression for Atlas 200I DK A2.

Single-shot inference wrapper around weight_regressor_fp16.om:
  input:  1x3x224x224 float32 (RGB, ImageNet-normalized)
  output: 1x1 float32 (weight_kg)

API
===
  m = WeightOMModel("/home/.../weight_regressor_fp16.om")
  kg = m.predict(roi_bgr)          # np.ndarray HxWx3 uint8 (BGR from cv2)

Assumes ACL already initialized by another model (e.g. NPUDetector). If not,
acl.init() is attempted but failures (already initialized) are tolerated.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_SUCCESS = 0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class WeightOMModel:
    """Pig weight regression on Ascend NPU. predict(roi_bgr) -> kg."""

    def __init__(self, om_path: str, input_size: int = 224, device_id: int = 0):
        self.input_size = input_size
        self.device_id = device_id

        # Import acl, falling back to common Ascend toolkit paths
        try:
            import acl
        except ImportError:
            for p in (
                "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
                "/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages",
            ):
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)
            import acl
        self.acl = acl

        # Tolerant init: if NPUDetector already called acl.init / set_device,
        # these will fail with a non-zero return; we ignore.
        try:
            acl.init()
        except Exception:
            pass
        try:
            acl.rt.set_device(device_id)
        except Exception:
            pass

        # Own context so we can re-bind from any thread
        self.context, ret = acl.rt.create_context(device_id)
        assert ret == ACL_SUCCESS, f"create_context failed: {ret}"

        self.model_id, ret = acl.mdl.load_from_file(om_path)
        assert ret == ACL_SUCCESS, f"load_from_file failed: {ret} for {om_path}"

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        assert ret == ACL_SUCCESS, f"get_desc failed: {ret}"

        self._setup_io()
        print(f"[WeightOM] Loaded: {om_path}")

    def _setup_io(self):
        acl = self.acl

        self.input_dataset = acl.mdl.create_dataset()
        self.input_buffers = []
        n_in = acl.mdl.get_num_inputs(self.model_desc)
        for i in range(n_in):
            size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            buf, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            assert ret == ACL_SUCCESS
            data = acl.create_data_buffer(buf, size)
            acl.mdl.add_dataset_buffer(self.input_dataset, data)
            self.input_buffers.append((buf, size))

        self.output_dataset = acl.mdl.create_dataset()
        self.output_buffers = []
        self.output_sizes = []
        n_out = acl.mdl.get_num_outputs(self.model_desc)
        for i in range(n_out):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            buf, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            assert ret == ACL_SUCCESS
            data = acl.create_data_buffer(buf, size)
            acl.mdl.add_dataset_buffer(self.output_dataset, data)
            self.output_buffers.append((buf, size))
            self.output_sizes.append(size)

        # Pre-allocate pinned host buffer for D2H (output is tiny: 4 bytes)
        out_size = self.output_sizes[0]
        self.host_out_buf, ret = acl.rt.malloc_host(out_size)
        assert ret == ACL_SUCCESS, f"malloc_host failed: {ret}"
        self.host_out_size = out_size

    def preprocess(self, roi_bgr: np.ndarray) -> np.ndarray:
        """BGR HxWx3 uint8 -> 1x3x224x224 float32 (RGB, ImageNet-normalized)."""
        if roi_bgr.size == 0:
            raise ValueError("Empty ROI")
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_LINEAR)
        chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(chw[np.newaxis, ...])

    def predict(self, roi_bgr: np.ndarray) -> float:
        """Return weight in kg. ROI is a BGR crop (HxWx3 uint8)."""
        blob = self.preprocess(roi_bgr)

        acl = self.acl
        acl.rt.set_context(self.context)

        # H2D
        buf, size = self.input_buffers[0]
        src_bytes = blob.tobytes()
        src_ptr = acl.util.bytes_to_ptr(src_bytes)
        ret = acl.rt.memcpy(buf, size, src_ptr, len(src_bytes),
                            ACL_MEMCPY_HOST_TO_DEVICE)
        assert ret == ACL_SUCCESS, f"H2D failed: {ret}"

        # Inference
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        assert ret == ACL_SUCCESS, f"execute failed: {ret}"

        # D2H
        buf, size = self.output_buffers[0]
        ret = acl.rt.memcpy(self.host_out_buf, self.host_out_size,
                            buf, size, ACL_MEMCPY_DEVICE_TO_HOST)
        assert ret == ACL_SUCCESS, f"D2H failed: {ret}"
        out_bytes = acl.util.ptr_to_bytes(self.host_out_buf, size)
        out = np.frombuffer(out_bytes, dtype=np.float32)
        return float(out.reshape(-1)[0])

    def __del__(self):
        try:
            acl = self.acl
            if getattr(self, "host_out_buf", None):
                acl.rt.free_host(self.host_out_buf)
            acl.mdl.unload(self.model_id)
            acl.mdl.destroy_desc(self.model_desc)
            for buf, _ in self.input_buffers:
                acl.rt.free(buf)
            for buf, _ in self.output_buffers:
                acl.rt.free(buf)
            acl.mdl.destroy_dataset(self.input_dataset)
            acl.mdl.destroy_dataset(self.output_dataset)
            acl.rt.destroy_context(self.context)
            # NOTE: we deliberately do NOT call reset_device / finalize here
            # because the parent detector may still own the device.
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--om", required=True, help=".om model path")
    ap.add_argument("--image", help="single RGB image to test on")
    args = ap.parse_args()

    m = WeightOMModel(args.om)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"cannot read {args.image}")
        kg = m.predict(img)
        print(f"[predict] {args.image} -> {kg:.2f} kg")
    else:
        # synthetic warm-up + benchmark
        import time
        synth = np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8)
        for _ in range(3):
            m.predict(synth)
        N = 50
        t0 = time.time()
        for _ in range(N):
            _ = m.predict(synth)
        dt = (time.time() - t0) / N * 1000
        print(f"[bench] {dt:.1f} ms/inference ({1000/dt:.1f} FPS)")
