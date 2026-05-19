import cv2
import numpy as np
import onnxruntime
import torch
from typing import Tuple, List, Dict, Union
import math

def letterbox(img: np.ndarray,
              new_shape: Tuple[int, int] = (640, 640),
              color: Tuple[int, int, int] = (114, 114, 114),
              auto: bool = True,
              scaleFill: bool = False,
              scaleup: bool = True,
              stride: int = 32) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """调整图像大小并保持宽高比，同时进行填充。
    
    Args:
        img: 输入图像
        new_shape: 目标尺寸 (height, width)
        color: 填充颜色
        auto: 是否自动调整最小填充
        scaleFill: 是否拉伸填充
        scaleup: 是否允许放大
        stride: 步长
    
    Returns:
        img: 处理后的图像
        ratio: 缩放比例
        (dw, dh): 填充尺寸
    """
    shape = img.shape[:2]  # 当前尺寸 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不放大
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # 宽度，高度缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度，高度缩放比例

    dw, dh = dw / 2, dh / 2  # 填充到上下左右
    if shape[::-1] != new_unpad:  # 调整大小
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return img, ratio[0], (dw, dh)

def preprocess_image(img_path: str, img_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """预处理图像。
    
    Args:
        img_path: 图像路径
        img_size: 目标尺寸 (height, width)
    
    Returns:
        img: 预处理后的图像
        ratio: 缩放比例
        (dw, dh): 填充尺寸
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    # BGR转RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整大小和填充
    img, ratio, (dw, dh) = letterbox(img, img_size, auto=False)
    
    # 转换为float32并归一化
    img = img.astype(np.float32) / 255.0
    
    # 转换为NCHW格式
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    
    return img, ratio, (dw, dh)

def postprocess_output(outputs: List[np.ndarray], 
                      ratio: float, 
                      pad: Tuple[float, float],
                      conf_thres: float = 0.25,
                      iou_thres: float = 0.45) -> List[Dict]:
    """处理模型输出。
    
    Args:
        outputs: 模型输出
        ratio: 缩放比例
        pad: 填充尺寸 (dw, dh)
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
    
    Returns:
        detections: 检测结果列表
    """
    num_dets, boxes, scores, labels = outputs
    
    # 获取有效检测结果
    num_dets = int(num_dets[0])
    boxes = boxes[0, :num_dets]
    scores = scores[0, :num_dets]
    labels = labels[0, :num_dets]
    
    # 转换坐标
    dw, dh = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    
    # 过滤低置信度检测结果
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # 执行NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    
    detections = []
    for i in indices:
        detections.append({
            'box': boxes[i].tolist(),
            'score': float(scores[i]),
            'label': int(labels[i])
        })
    
    return detections

def main():
    single_image = False

    # 模型和图像路径
    model_path = "/home/xyzt2204/project/mmyolo/mmyolo/work_dirs/onnx_export/export_oakdetv2/epoch_230.onnx"
    image_dir = "/home/xyzt2204/project/mmyolo/mmyolo/data/holedet/oak/oak_test/images"

    # 创建ONNX运行时会话，尝试使用GPU
    try:
        # 首先尝试使用CUDA
        session = onnxruntime.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print("使用GPU (CUDA) 进行推理")
    except Exception as e:
        print(f"无法使用GPU: {e}")
        print("使用CPU进行推理")
        session = onnxruntime.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
    if not single_image:
        import os   
        detections_list = []
        image_paths = []
        for img_path in os.listdir(image_dir):
            full_img_path = os.path.join(image_dir, img_path)
            img, ratio, pad = preprocess_image(full_img_path)
            outputs = session.run(None, {'images': img})
            detections = postprocess_output(outputs, ratio, pad)
            print("finish image: ", full_img_path)
            detections_list.append(detections)
            image_paths.append(img_path)
        
        

        
        # 打印结果
        for i, (detections, img_name) in enumerate(zip(detections_list, image_paths), 1):
            print(f"\n第{i}张图片的检测结果 (文件名: {img_name}):")
            for j, det in enumerate(detections, 1):
                print(f"目标 {j}:")
                print(f"  类别: {det['label']}")
                print(f"  置信度: {det['score']:.4f}")
                print(f"  边界框: x1={det['box'][0]:.1f}, y1={det['box'][1]:.1f}, x2={det['box'][2]:.1f}, y2={det['box'][3]:.1f}")
                print()
    else:
        img_path = "/home/xyzt2204/project/holedet/oak/oak_datasetv1/images/frame_000045.jpg"
        img, ratio, pad = preprocess_image(img_path)
        outputs = session.run(None, {'images': img})
        detections = postprocess_output(outputs, ratio, pad)
        print(detections)

if __name__ == "__main__":
    main()
