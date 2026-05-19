# OAK 坑洞检测（MMYOLO）

基于 [OpenMMLab MMYOLO](https://github.com/open-mmlab/mmyolo) 的 **路面坑洞（hole）单类别目标检测** 工程。在 MMYOLO 框架上增加了 OAK 数据集配置、自定义 Anchor 与 YOLOv5 / YOLOv8 训练配置。

> 上游 MMYOLO 原版说明见：[英文](docs/README_mmyolo_original_en.md) | [简体中文](docs/README_mmyolo_original_zh-CN.md)

## 功能概览

| 内容 | 说明 |
|------|------|
| 检测类别 | `hole`（单类） |
| 数据格式 | COCO JSON + `images/` 目录 |
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

## 数据集结构

数据集独立于本仓库，默认放在 `holedet/oak/` 下（与训练机路径一致时可修改配置中的 `data_root`）。

### 目录布局（以 `oak_datasetv2` 为例）

```text
oak_datasetv2/
├── images/                 # 训练/验证/测试图片（如 frame_000483.jpg）
├── annotations/
│   ├── trainval.json       # 训练+验证，COCO 格式
│   ├── test.json           # 测试集，COCO 格式
│   └── result.json         # 可选：推理或导出结果
└── labels/                 # 可选：YOLO 格式 txt，便于转换或检查
```

### COCO 标注要点

- `images[].file_name` 对应 `images/` 下的文件名。
- `annotations` 中 `category_id` 为坑洞框；`metainfo` 中类别名为 `hole`。
- 配置中通过 `data_prefix=dict(img='images/')` 指定图片子目录。

### 数据集版本

| 目录 | 说明 | 配置中默认用途 |
|------|------|----------------|
| `oak_datasetv1/` | 早期版本，578 张图 | `oak_hole.py`、`oak_hole_anchors.py` |
| `oak_datasetv2/` | 修订版本，578 张图 | `oak_hole_anchors_v2.py`、`oak_yolov8_s_fast_1xb12-40e_hole.py` |

### 配置路径修改

配置内 `data_root` 现为 Linux 训练机绝对路径，克隆到本机后请改为本地路径，例如：

```python
# Windows 示例
data_root = 'E:/4070project/holedet/oak/oak_datasetv2/'
```

## 配置文件说明

所有坑洞相关配置位于 `configs/yolov5/` 与 `configs/yolov8/`。

| 配置文件 | 基座模型 | 数据集 | 说明 |
|----------|----------|--------|------|
| `configs/yolov5/oak_hole.py` | YOLOv5-s | v1 | 小目标 Anchor，可 `load_from` 断点续训 |
| `configs/yolov5/oak_hole_anchors.py` | YOLOv5-s | v1 | v1 上 K-Means 优化后的 Anchor |
| `configs/yolov5/oak_hole_anchors_v2.py` | YOLOv5-s | v2 | v2 优化 Anchor（推荐 YOLOv5 实验） |
| `configs/yolov5/yolov5_s-v61_fast_1xb12-40e_hole.py` | YOLOv5-s | `camera/` | 相机采集子集实验 |
| `configs/yolov8/oak_yolov8_s_fast_1xb12-40e_hole.py` | YOLOv8-s | v2 | 关闭部分 heavy aug 的 fast 配置 |

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
# 单卡训练（示例：YOLOv5 + v2）
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

以下内容 **不纳入 Git**，已打包或计划打包至 `E:\4070project\mmyolo\baidupan\`，上传百度网盘后请在 Issue / 文档中补充分享链接。

| 压缩包 | 内容 |
|--------|------|
| `mmyolo_work_dirs.tar.gz` | 训练权重、日志、`vis_data` 等 |
| `oak_datasetv1.tar.gz` | 数据集 v1 |
| `oak_datasetv2.tar.gz` | 数据集 v2 |

清单与解压说明见：[docs/BAIDU_PAN_ASSETS.md](docs/BAIDU_PAN_ASSETS.md)。

本地打包可执行：

```powershell
.\scripts\pack_baidupan_assets.ps1
```

## 仓库结构（与本项目相关部分）

```text
mmyolo/
├── configs/yolov5/          # OAK 坑洞 YOLOv5 配置
├── configs/yolov8/          # OAK 坑洞 YOLOv8 配置
├── demo/                    # 推理脚本
├── docs/                    # 文档（含 MMYOLO 原版 README）
├── scripts/                 # 辅助脚本（网盘打包等）
├── tools/                   # 训练 / 测试 / 分析工具
├── work_dirs/               # 本地实验输出（gitignore）
└── vis_res/                 # 可视化结果（gitignore）
```

## 引用

若使用 MMYOLO，请引用 OpenMMLab 官方论文与仓库（见 [docs/README_mmyolo_original_en.md](docs/README_mmyolo_original_en.md) 中的 Citation 章节）。

## 上传 GitHub

本目录已重新 `git init`，大文件目录已由 `.gitignore` 排除。首次提交示例：

```bash
cd E:/4070project/mmyolo/mmyolo
git add .
git commit -m "feat: OAK hole detection configs and docs"
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

若在 Windows 上遇到 `dubious ownership`，可对当前仓库单独放行（勿改全局配置时可省略，改用管理员或统一账户克隆）：

```bash
git config --global --add safe.directory E:/4070project/mmyolo/mmyolo
```

## 许可证

本仓库基于 MMYOLO，遵循原项目 [Apache 2.0](LICENSE) 许可证。
