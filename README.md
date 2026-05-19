# OAK 坑洞检测（MMYOLO）

基于 [OpenMMLab MMYOLO](https://github.com/open-mmlab/mmyolo) 的 **路面坑洞（hole）单类别目标检测** 工程。在 MMYOLO 框架上增加了 OAK 数据集配置、自定义 Anchor 与 YOLOv5 / YOLOv8 训练配置。

> 上游 MMYOLO 原版说明见：[英文](docs/README_mmyolo_original_en.md) | [简体中文](docs/README_mmyolo_original_zh-CN.md)

**代码仓库：** https://github.com/Small-NengNeng/dethole

## 功能概览

| 内容 | 说明 |
|------|------|
| 检测类别 | `hole`（单类） |
| 数据格式 | COCO JSON + `images/`（训练配置）；网盘提供 YOLO 格式压缩包 |
| 主要模型 | YOLOv5-s（自定义 Anchor）、YOLOv8-s |
| 图像分辨率 | 3840×2160（4K 帧） |

## 环境安装

建议使用 Conda，并安装与 MMDetection 3.x / MMEngine 兼容的 PyTorch 与 MMCV。

```bash
cd mmyolo
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -v -e .
```

详细依赖与版本说明请参考 [MMYOLO 安装文档](https://mmyolo.readthedocs.io/en/latest/get_started/installation.html)。

## 数据集

训练/验证数据 **不纳入 Git**，请从百度网盘下载并解压到本地：

| 项目 | 说明 |
|------|------|
| 文件名 | `real_trainval_yolo.7z` |
| 链接 | https://pan.baidu.com/s/1x8ScM9siq7yv2JvgdL4mqg |
| 提取码 | `8ar2` |

建议解压路径：

```text
E:/4070project/holedet/oak/real_trainval_yolo.7z   # 压缩包（下载后）
E:/4070project/holedet/oak/<解压目录>/             # 解压后的数据集根目录
```

解压后目录通常包含 `images/` 与 `labels/`（YOLO 标注）。若使用本仓库中的 COCO 格式配置训练，请将 `data_root` 指向含 `images/` 与 `annotations/trainval.json`、`annotations/test.json` 的目录，或先将 YOLO 标注转换为 COCO（参见 `holedet/oak` 下相关脚本）。

### 配置路径修改

配置内 `data_root` 需改为本机解压后的数据集根目录，例如：

```python
data_root = 'E:/4070project/holedet/oak/real_trainval_yolo/'
```

## 配置文件说明

所有坑洞相关配置位于 `configs/yolov5/` 与 `configs/yolov8/`。

| 配置文件 | 基座模型 | 说明 |
|----------|----------|------|
| `configs/yolov5/oak_hole.py` | YOLOv5-s | 小目标 Anchor，可 `load_from` 断点续训 |
| `configs/yolov5/oak_hole_anchors.py` | YOLOv5-s | K-Means 优化 Anchor |
| `configs/yolov5/oak_hole_anchors_v2.py` | YOLOv5-s | 优化 Anchor（推荐 YOLOv5 实验） |
| `configs/yolov5/yolov5_s-v61_fast_1xb12-40e_hole.py` | YOLOv5-s | 相机采集子集实验 |
| `configs/yolov8/oak_yolov8_s_fast_1xb12-40e_hole.py` | YOLOv8-s | fast 训练配置 |

### Anchor 优化（YOLOv5）

```bash
python tools/analysis_tools/optimize_anchors.py \
  configs/yolov5/oak_hole_anchors_v2.py \
  --algorithm k-means \
  --input-shape 640 640 \
  --out-dir work_dirs/anchor_info/oak_v2_anchor_info
```

结果写入 `anchor_optimize_result.json`，再填入配置中的 `anchors` 列表。

## 训练与测试

在项目根目录（含 `configs/` 的目录）执行：

```bash
# 单卡训练（示例）
python tools/train.py configs/yolov5/oak_hole_anchors_v2.py

# 测试 / 评估
python tools/test.py configs/yolov5/oak_hole_anchors_v2.py work_dirs/oak_hole_anchors_v2/best_coco_bbox_mAP_epoch_*.pth

# 浏览数据集与标注
python tools/analysis_tools/browse_dataset.py configs/yolov5/oak_hole_anchors_v2.py
```

日志、权重与可视化默认写入 `work_dirs/<配置名>/`（已加入 `.gitignore`）。

### 推理演示

```bash
python demo/image_demo.py \
  path/to/image.jpg \
  configs/yolov5/oak_hole_anchors_v2.py \
  work_dirs/oak_hole_anchors_v2/best_coco_bbox_mAP_epoch_*.pth \
  --out-dir output/demo
```

## 大文件与网盘资源

以下内容 **不纳入 Git**，请从百度网盘获取：

| 资源 | 说明 |
|------|------|
| `real_trainval_yolo.7z` | 训练/验证数据集（[链接](https://pan.baidu.com/s/1x8ScM9siq7yv2JvgdL4mqg?pwd=8ar2)，提取码 `8ar2`） |
| `mmyolo_work_dirs.tar.gz` | 预训练权重、训练日志与实验输出（[链接](https://pan.baidu.com/s/1VIYYhSidqYELLCOrKkracg?pwd=ue4p)，提取码 `ue4p`） |

下载 `mmyolo_work_dirs.tar.gz` 后解压示例：

```powershell
# 假设压缩包保存在 baidupan 目录
tar -xzf E:\4070project\mmyolo\baidupan\mmyolo_work_dirs.tar.gz -C E:\4070project\mmyolo\mmyolo
```

更多说明见 [docs/BAIDU_PAN_ASSETS.md](docs/BAIDU_PAN_ASSETS.md)。

## 仓库结构（与本项目相关部分）

```text
mmyolo/
├── configs/yolov5/          # OAK 坑洞 YOLOv5 配置
├── configs/yolov8/          # OAK 坑洞 YOLOv8 配置
├── demo/                    # 推理脚本
├── docs/                    # 文档（含 MMYOLO 原版 README）
├── scripts/                 # 辅助脚本
├── tools/                   # 训练 / 测试 / 分析工具
├── work_dirs/               # 本地实验输出（gitignore）
└── vis_res/                 # 可视化结果（gitignore）
```

## 引用

若使用 MMYOLO，请引用 OpenMMLab 官方论文与仓库（见 [docs/README_mmyolo_original_en.md](docs/README_mmyolo_original_en.md) 中的 Citation 章节）。

## 许可证

本仓库基于 MMYOLO，遵循原项目 [Apache 2.0](LICENSE) 许可证。
