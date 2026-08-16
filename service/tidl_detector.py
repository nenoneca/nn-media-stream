"""YOLOX-s-lite on the TI C7x via onnxruntime's TIDLExecutionProvider.

Mirrors nn_infer's byai_ort C PAL exactly (same model dir, same pre/post):
  input  "images": uint8 NCHW {1,3,640,640}, BGR, caller already letterboxed
  output 0: float dets[1,N,5] = x1,y1,x2,y2,score   (model-input space)
  output 1: labels[1,N]                             (COCO80 ids)

The model dir is a TI model-zoo entry (ONR-OD-8220-...): the .onnx at the
top level, compiled artifacts in artifacts/.
"""

import glob
import os

import numpy as np


class TidlYoloxDetector:
    def __init__(self, model_dir: str, core: int = 1):
        import onnxruntime as ort
        onnx = (sorted(glob.glob(os.path.join(model_dir, "*.onnx"))) or
                sorted(glob.glob(os.path.join(model_dir, "model", "*.onnx"))))
        if not onnx:
            raise FileNotFoundError(f"no .onnx in {model_dir}(/model)")
        opts = {
            "artifacts_folder": os.path.join(model_dir, "artifacts"),
            "debug_level": 0,
            "core_number": core,
        }
        so = ort.SessionOptions()
        self.sess = ort.InferenceSession(
            onnx[0], sess_options=so,
            providers=["TIDLExecutionProvider"],
            provider_options=[opts])
        self.input = self.sess.get_inputs()[0].name
        self.outs = [o.name for o in self.sess.get_outputs()]

    def __call__(self, rgb: np.ndarray, min_score: float = 0.45) -> list:
        # RGB HWC -> BGR CHW uint8 (the PAL feeds BGR; artifacts were
        # compiled with reverse_channels)
        chw = np.ascontiguousarray(
            rgb[..., ::-1].transpose(2, 0, 1)[None])
        res = self.sess.run(self.outs[:2], {self.input: chw})
        dets, labels = res[0].reshape(-1, 5), res[1].reshape(-1)
        out = []
        for i in range(dets.shape[0]):
            sc = float(dets[i, 4])
            if sc < min_score:
                continue
            x1, y1, x2, y2 = (float(v) for v in dets[i, :4])
            if x2 <= x1 or y2 <= y1:
                continue
            out.append({"class_id": int(labels[i]),
                        "score": round(sc, 3),
                        "box": [int(x1), int(y1),
                                int(x2 - x1), int(y2 - y1)]})
        return out
