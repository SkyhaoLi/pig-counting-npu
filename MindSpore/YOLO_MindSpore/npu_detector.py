"""
NPU inference using ACL for YOLOv8n pig detection on Atlas 200I DK A2.
Uses the .om model format for Ascend 310B4 NPU acceleration.
"""
import numpy as np
import cv2
import time
import os
import sys
import struct

# ACL constants
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_SUCCESS = 0

class NPUDetector:
    """YOLOv8 detector using Ascend NPU via ACL."""

    def __init__(self, om_path, conf_thres=0.25):
        self.conf_thres = conf_thres
        self.input_size = 640

        # Try to import acl
        try:
            import acl
            self.acl = acl
        except ImportError:
            # Try adding ACL python path
            acl_paths = [
                '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages',
                '/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages',
                '/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64',
            ]
            for p in acl_paths:
                if os.path.exists(p) and p not in sys.path:
                    sys.path.insert(0, p)
            try:
                import acl
                self.acl = acl
            except ImportError:
                raise ImportError("Cannot import acl. Make sure CANN toolkit is installed.")

        # Init ACL
        ret = self.acl.init()
        assert ret == ACL_SUCCESS, f"acl.init failed: {ret}"

        ret = self.acl.rt.set_device(0)
        assert ret == ACL_SUCCESS, f"set_device failed: {ret}"

        self.context, ret = self.acl.rt.create_context(0)
        assert ret == ACL_SUCCESS, f"create_context failed: {ret}"

        self.stream, ret = self.acl.rt.create_stream()
        assert ret == ACL_SUCCESS, f"create_stream failed: {ret}"

        # Load model
        self.model_id, ret = self.acl.mdl.load_from_file(om_path)
        assert ret == ACL_SUCCESS, f"load model failed: {ret}"

        self.model_desc = self.acl.mdl.create_desc()
        ret = self.acl.mdl.get_desc(self.model_desc, self.model_id)
        assert ret == ACL_SUCCESS, f"get_desc failed: {ret}"

        # Setup input/output datasets
        self._setup_io()
        print(f"[NPU] Loaded: {om_path}")

    def _setup_io(self):
        acl = self.acl

        # Input
        self.input_dataset = acl.mdl.create_dataset()
        n_inputs = acl.mdl.get_num_inputs(self.model_desc)
        self.input_buffers = []
        for i in range(n_inputs):
            size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            buf, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            assert ret == ACL_SUCCESS
            data = acl.create_data_buffer(buf, size)
            acl.mdl.add_dataset_buffer(self.input_dataset, data)
            self.input_buffers.append((buf, size))

        # Output
        self.output_dataset = acl.mdl.create_dataset()
        n_outputs = acl.mdl.get_num_outputs(self.model_desc)
        self.output_buffers = []
        self.output_sizes = []
        for i in range(n_outputs):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            buf, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            assert ret == ACL_SUCCESS
            data = acl.create_data_buffer(buf, size)
            acl.mdl.add_dataset_buffer(self.output_dataset, data)
            self.output_buffers.append((buf, size))
            self.output_sizes.append(size)

    def preprocess(self, img):
        """Same preprocessing as ONNX detector."""
        h, w = img.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        nh, nw = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (nw, nh))

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = img_resized

        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return np.ascontiguousarray(blob), scale

    def detect(self, img):
        """Run detection on a single image, return list of [x1,y1,x2,y2,conf,cls]."""
        blob, scale = self.preprocess(img)

        acl = self.acl

        # Copy input to device
        buf, size = self.input_buffers[0]
        blob_contig = np.ascontiguousarray(blob)
        blob_ptr = acl.util.numpy_to_ptr(blob_contig)
        blob_size = blob_contig.nbytes
        ret = acl.rt.memcpy(buf, size, blob_ptr, blob_size, ACL_MEMCPY_HOST_TO_DEVICE)
        assert ret == ACL_SUCCESS, f"memcpy H2D failed: {ret}"

        # Run inference
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        assert ret == ACL_SUCCESS, f"execute failed: {ret}"

        # Get output - first output is [1, 8400, 5] (boxes + objectness)
        buf, size = self.output_buffers[0]
        output_data = np.zeros(size // 4, dtype=np.float32)  # float32
        output_ptr = acl.util.numpy_to_ptr(output_data)
        ret = acl.rt.memcpy(output_ptr, size, buf, size, ACL_MEMCPY_DEVICE_TO_HOST)
        assert ret == ACL_SUCCESS, f"memcpy D2H failed: {ret}"

        # Reshape to [8400, 5]
        output = output_data.reshape(1, 8400, 5)[0]

        # Parse detections (same as ONNX detector)
        # columns: cx, cy, w, h, conf (single class)
        conf = output[:, 4]
        mask = conf > self.conf_thres
        dets = output[mask]

        if len(dets) == 0:
            return np.empty((0, 6))

        cx, cy, w, h = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        x1 = (cx - w / 2) / scale
        y1 = (cy - h / 2) / scale
        x2 = (cx + w / 2) / scale
        y2 = (cy + h / 2) / scale
        scores = dets[:, 4]

        # Apply NMS
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = self._nms(boxes, scores, iou_thres=0.45)

        x1, y1, x2, y2, scores = x1[keep], y1[keep], x2[keep], y2[keep], scores[keep]
        results = np.stack([x1, y1, x2, y2, scores, np.zeros_like(scores)], axis=1)
        return results

    @staticmethod
    def _nms(boxes, scores, iou_thres=0.45):
        """Pure numpy NMS."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=int)

    def __del__(self):
        try:
            acl = self.acl
            acl.mdl.unload(self.model_id)
            acl.mdl.destroy_desc(self.model_desc)
            # Free buffers
            for buf, _ in self.input_buffers:
                acl.rt.free(buf)
            for buf, _ in self.output_buffers:
                acl.rt.free(buf)
            acl.mdl.destroy_dataset(self.input_dataset)
            acl.mdl.destroy_dataset(self.output_dataset)
            acl.rt.destroy_stream(self.stream)
            acl.rt.destroy_context(self.context)
            acl.rt.reset_device(0)
            acl.finalize()
        except:
            pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--om', required=True, help='OM model path')
    parser.add_argument('--video', required=True, help='Video path')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()

    det = NPUDetector(args.om, args.conf)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    times = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        dets = det.detect(frame)
        t1 = time.time()
        times.append((t1 - t0) * 1000)
        count += 1

        if count <= 3 or count % 50 == 0:
            print(f"Frame {count}/{total}: {len(dets)} detections, {times[-1]:.1f}ms")

    cap.release()

    if times:
        avg = np.mean(times[1:])  # skip first frame (warmup)
        print(f"\nTotal: {count} frames, avg {avg:.1f}ms/frame ({1000/avg:.1f} FPS)")
