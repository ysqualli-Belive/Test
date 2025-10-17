#!/usr/bin/env python3
"""
Crop detections from images using an ONNX object detection model.

Defaults are set to your paths:
- --model "C:\Users\User\Desktop\A\etiquette.onnx"
- --input "\\192.168.20.50\Dev\Baneasa - OCR - 0910"
- --output "C:\Users\User\Desktop\A\Res"

You can override via CLI flags.
"""

import argparse
import os
import sys
import glob
from pathlib import Path
import numpy as np

try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV (cv2) is required. Install with: pip install opencv-python", file=sys.stderr)
    raise

try:
    import onnxruntime as ort
except Exception as e:
    print("ERROR: onnxruntime is required. Install with: pip install onnxruntime", file=sys.stderr)
    raise


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), stride=32, auto=False, scaleup=True):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (left, top)


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms_numpy(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def parse_model_output(output_list, conf_thres=0.25, iou_thres=0.5, input_shape=(640, 640), pad=(0,0), gain=(1.0,1.0), img_shape=None):
    dets = []
    outs = [np.asarray(o) for o in output_list]
    outs.sort(key=lambda a: a.size, reverse=True)
    out = np.squeeze(outs[0])

    def scale_coords(boxes_xyxy):
        if img_shape is None:
            return boxes_xyxy
        h0, w0 = img_shape[:2]
        x_pad, y_pad = pad
        gx, gy = gain
        boxes = boxes_xyxy.copy()
        boxes[:, [0,2]] = (boxes[:, [0,2]] - x_pad) / gx
        boxes[:, [1,3]] = (boxes[:, [1,3]] - y_pad) / gy
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w0 - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w0 - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h0 - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h0 - 1)
        return boxes

    if out.ndim == 2 and out.shape[1] >= 6:
        if out.shape[1] == 6:
            boxes = out[:, :4].astype(np.float32)
            scores = out[:, 4].astype(np.float32)
            clss = out[:, 5].astype(np.int32)
            boxes = scale_coords(boxes)
            final = []
            for c in np.unique(clss):
                m = clss == c
                keep = nms_numpy(boxes[m], scores[m], iou_threshold=iou_thres)
                for i in keep:
                    if scores[m][i] >= conf_thres:
                        final.append((boxes[m][i], float(scores[m][i]), int(c)))
            return final
        elif out.shape[1] > 6:
            xywh = out[:, 0:4].astype(np.float32)
            obj = out[:, 4].astype(np.float32)
            cls_scores = out[:, 5:].astype(np.float32)
            if cls_scores.ndim == 2 and cls_scores.shape[1] > 0:
                cls_ids = np.argmax(cls_scores, axis=1)
                cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
                scores = obj * cls_conf
            else:
                cls_ids = np.zeros_like(obj, dtype=np.int32)
                scores = obj
            boxes = xywh2xyxy(xywh)
            boxes = scale_coords(boxes)
            final = []
            for c in np.unique(cls_ids):
                m = cls_ids == c
                if np.sum(m) == 0:
                    continue
                keep = nms_numpy(boxes[m], scores[m], iou_threshold=iou_thres)
                for i in keep:
                    if scores[m][i] >= conf_thres:
                        final.append((boxes[m][i], float(scores[m][i]), int(c)))
            return final

    if out.ndim == 2 and out.shape[1] == 7:
        clss = out[:, 1].astype(np.int32)
        scores = out[:, 2].astype(np.float32)
        boxes = out[:, 3:7].astype(np.float32)
        if boxes.max() <= 1.2:
            iw, ih = input_shape[1], input_shape[0]
            scale = np.array([iw, ih, iw, ih], dtype=np.float32)
            boxes = boxes * scale
        boxes = scale_coords(boxes)
        final = []
        for c in np.unique(clss):
            m = clss == c
            keep = nms_numpy(boxes[m], scores[m], iou_threshold=iou_thres)
            for i in keep:
                if scores[m][i] >= conf_thres:
                    final.append((boxes[m][i], float(scores[m][i]), int(c)))
        return final

    return dets


def load_image(path):
    im = cv2.imread(str(path))
    if im is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return im


def preprocess(im, input_hw):
    h, w = input_hw
    img, ratio, pad = letterbox(im, (h, w), auto=False)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    img_b = np.expand_dims(img_chw, 0)  # [1,3,H,W]
    return img_b, ratio, pad, img.shape[:2]


def main():
    default_model = r"C:\Users\User\Desktop\A\etiquette.onnx"
    default_input = r"\\192.168.20.50\Dev\Baneasa - OCR - 0910"
    default_output = r"C:\Users\User\Desktop\A\Res"

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=default_model, help="Path to .onnx file")
    ap.add_argument("--input", type=str, default=default_input, help="Input folder with images")
    ap.add_argument("--output", type=str, default=default_output, help="Output folder for cropped detections")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="IOU threshold for NMS")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    providers = ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(args.model, providers=providers)
    except Exception as e:
        print(f"ERROR: Failed to load ONNX model: {e}", file=sys.stderr)
        sys.exit(1)

    input_meta = sess.get_inputs()[0]
    inp_name = input_meta.name
    inp_shape = list(input_meta.shape)
    if len(inp_shape) == 4:
        H = int(inp_shape[2]) if isinstance(inp_shape[2], (int, np.integer)) else 640
        W = int(inp_shape[3]) if isinstance(inp_shape[3], (int, np.integer)) else 640
    else:
        H, W = 640, 640
    input_hw = (H, W)
    print(f"Model input: name={inp_name}, shape={inp_shape} -> using size {input_hw}")

    files = []
    for p in glob.glob(str(in_dir / "**" / "*"), recursive=True):
        if Path(p).suffix.lower() in IMG_EXTS:
            files.append(Path(p))
    files.sort()

    if not files:
        print(f"No images found in: {in_dir}")
        sys.exit(0)

    saved = 0
    for img_path in files:
        try:
            im0 = load_image(img_path)
        except Exception as e:
            print(f"[WARN] Skipping {img_path}: {e}", file=sys.stderr)
            continue

        img_b, ratio, pad, net_hw = preprocess(im0, input_hw)

        ort_inputs = {inp_name: img_b}
        outputs = sess.run(None, ort_inputs)

        r = min(input_hw[0] / im0.shape[0], input_hw[1] / im0.shape[1])
        new_w = int(round(im0.shape[1] * r))
        new_h = int(round(im0.shape[0] * r))
        dw = (input_hw[1] - new_w) / 2
        dh = (input_hw[0] - new_h) / 2

        dets = parse_model_output(outputs, conf_thres=args.conf, iou_thres=args.iou,
                                  input_shape=input_hw, pad=(dw, dh), gain=(r, r), img_shape=im0.shape)

        if not dets:
            print(f"[INFO] No detections in {img_path.name}")
            continue

        base = img_path.stem
        for idx, (box, score, cls_id) in enumerate(dets):
            x1, y1, x2, y2 = box.astype(int)
            crop = im0[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out_name = f"{base}_{idx:02d}_c{cls_id}_s{score:.2f}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), crop)
            saved += 1

        print(f"[OK] {img_path.name}: saved {len(dets)} crops")

    print(f"Done. Total crops saved: {saved}. Output folder: {out_dir}")


if __name__ == "__main__":
    main()
