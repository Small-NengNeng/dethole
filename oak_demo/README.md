# oak_demo

OAK 深度相机 + **MMYOLO 坑洞检测 ONNX** 的采集与实时推理演示。代码自 `oak-win/record_data` 迁移并整理。

> **说明**：本目录 **不包含 MMYOLO 训练逻辑**；训练请在仓库根目录使用 `tools/train.py`。此处负责相机取流、数据录制，以及 **ONNX Runtime 主机端推理**。

## 功能确认

| 能力 | 是否实现 | 脚本 |
|------|----------|------|
| OAK RGB + 深度取流 | 是 | `record_only.py`、`dethole_host.py` |
| 4K 帧录制（供标注/训练） | 是 | `record_only.py --save` |
| MMYOLO 导出 ONNX 离线推理 | 是 | `singleinfer_onnx.py` |
| 实时 ONNX 推理 + 深度测距 | 是 | `dethole_host.py` |
| 设备端 Blob 推理（MobileNet-SSD） | 否（未迁移） | 原 `dethole.py` 为旧方案 |

**ONNX 推理**：使用 `onnxruntime` 加载 MMYOLO EasyDeploy 导出的 YOLO ONNX（输入名 `images`，输出含 `num_dets/boxes/scores/labels`），与 `projects/easydeploy/tools/singleinfer_onnx.py` 后处理逻辑一致。

## 重要文件

| 文件 | 作用 |
|------|------|
| `singleinfer_onnx.py` | letterbox、预处理、NMS 后处理；离线单张/目录推理 |
| `dethole_host.py` | **主程序**：OAK 4K 视频流 + 主机 ONNX 检测 + 深度 XYZ |
| `record_only.py` | 仅采集/预览，不做推理 |

## 环境

```bash
cd oak_demo
pip install -r requirements.txt
```

需连接 OAK 设备；Windows 上请安装 DepthAI 官方 USB 驱动。

## ONNX 模型准备

1. 在 mmyolo 根目录训练，例如：

   ```bash
   python tools/train.py configs/yolov5/oak_hole_anchors_v2.py
   ```

2. 使用 EasyDeploy 导出 ONNX（示例）：

   ```bash
   python projects/easydeploy/tools/export_onnx.py \
     configs/yolov5/oak_hole_anchors_v2.py \
     work_dirs/oak_hole_anchors_v2/best_coco_bbox_mAP_epoch_*.pth \
     --work-dir oak_demo/onnx
   ```

3. 或将网盘权重解压后，将 `.onnx` 复制到 `oak_demo/onnx/`。

## 使用

### 1. 采集数据（无推理）

```bash
python record_only.py --rgb --save
```

帧保存至 `savedata/rgb_<时间戳>/`。

### 2. 实时检测 + 测距

```bash
python dethole_host.py --model onnx/epoch_230.onnx --conf 0.5
```

可选 `--save` 同时保存 4K 原图。按 `q` 退出。

### 3. 离线 ONNX 推理

```bash
python singleinfer_onnx.py --model onnx/epoch_230.onnx --image path/to.jpg
python singleinfer_onnx.py --model onnx/epoch_230.onnx --image-dir path/to/images
```

## 与 MMYOLO 训练流程的关系

```text
OAK 采集 (record_only) → 标注 → MMYOLO 训练 (tools/train.py)
                              → 导出 ONNX → dethole_host / singleinfer_onnx 推理
```

## 来源

迁移自 `E:\4070project\oak\oak-win\record_data\dethole\`，已去除硬编码的 `E:\project\...` 路径，默认使用本目录下 `onnx/` 与 `savedata/`。
