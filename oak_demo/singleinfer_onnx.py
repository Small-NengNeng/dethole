"""ONNX inference utilities for MMYOLO-exported hole detection models."""

import os

import cv2
import numpy as np
import onnxruntime
from pathlib import Path
from typing import Dict, List, Tuple, Union

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'onnx' / 'epoch_230.onnx'


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio[0], (dw, dh)


def preprocess_image(
    img_path: Union[str, Path],
    img_size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f'无法读取图像: {img_path}')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, ratio, pad = letterbox(img, img_size, auto=False)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img, ratio, pad


def create_onnx_session(model_path: Union[str, Path]) -> onnxruntime.InferenceSession:
    model_path = str(model_path)
    try:
        session = onnxruntime.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        print('使用 GPU (CUDA) 进行推理')
    except Exception as e:
        print(f'无法使用 GPU: {e}')
        print('使用 CPU 进行推理')
        session = onnxruntime.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
    return session


def postprocess_output(
    outputs: List[np.ndarray],
    ratio: float,
    pad: Tuple[float, float],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> List[Dict]:
    num_dets, boxes, scores, labels = outputs

    num_dets = int(num_dets[0])
    boxes = boxes[0, :num_dets]
    scores = scores[0, :num_dets]
    labels = labels[0, :num_dets]

    dw, dh = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio

    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), conf_thres, iou_thres)

    detections = []
    if len(indices) == 0:
        return detections
    for i in np.array(indices).reshape(-1):
        detections.append({
            'box': boxes[i].tolist(),
            'score': float(scores[i]),
            'label': int(labels[i]),
        })
    return detections


def infer_image(
    session: onnxruntime.InferenceSession,
    img_path: Union[str, Path],
    conf_thres: float = 0.25,
) -> List[Dict]:
    img, ratio, pad = preprocess_image(img_path)
    outputs = session.run(None, {'images': img})
    return postprocess_output(outputs, ratio, pad, conf_thres=conf_thres)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Offline ONNX hole detection')
    parser.add_argument(
        '--model',
        type=str,
        default=str(DEFAULT_MODEL),
        help='Path to ONNX model (MMYOLO easydeploy export)',
    )
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image-dir', type=str, help='Directory of images')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()

    session = create_onnx_session(args.model)

    if args.image:
        dets = infer_image(session, args.image, args.conf)
        print(dets)
    elif args.image_dir:
        for name in sorted(os.listdir(args.image_dir)):
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = Path(args.image_dir) / name
            dets = infer_image(session, path, args.conf)
            print(f'{name}: {len(dets)} detections')
    else:
        parser.error('请指定 --image 或 --image-dir')


if __name__ == '__main__':
    main()
