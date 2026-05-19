# no_learning_app

无学习/传统图像处理流水线脚本（孔洞检测相关）。

## 目录说明

本仓库仅包含 Python 源码（`.py`）。运行时在本地生成的输出目录（如 `camera_vis/`、`result/`、`grey/`、`img/`、`mask/`、`vis_mask/` 及各类图像文件）已写入 `.gitignore`，不会纳入版本库。

## 脚本

| 文件 | 说明 |
|------|------|
| `rgb2grey.py` | RGB 转灰度 |
| `analygrey.py` | 灰度分析 |
| `gettarget.py` | 目标提取 |
| `box.py` | 框选/几何处理 |
| `batch_process.py` | 批处理入口 |
