# 百度网盘大文件说明

GitHub 仓库仅包含代码与配置，下列资源请从百度网盘下载后按说明解压。

## 压缩包位置

默认生成目录：`E:\4070project\mmyolo\baidupan\`

| 文件名 | 源路径 | 说明 |
|--------|--------|------|
| `mmyolo_work_dirs.tar.gz` | `mmyolo/work_dirs/` | 训练 checkpoint（`.pth`）、日志、TensorBoard/vis 数据、ONNX 导出等 |
| `oak_datasetv1.tar.gz` | `holedet/oak/oak_datasetv1/` | 坑洞数据集 v1 |
| `oak_datasetv2.tar.gz` | `holedet/oak/oak_datasetv2/` | 坑洞数据集 v2 |

> **网盘链接**：上传后请在本文件或根目录 `README.md` 中填写分享链接与提取码。

## 解压示例

```powershell
# 训练产物 → 放回 mmyolo 工程
tar -xzf E:\4070project\mmyolo\baidupan\mmyolo_work_dirs.tar.gz -C E:\4070project\mmyolo\mmyolo

# 数据集 → holedet 目录
New-Item -ItemType Directory -Force -Path E:\4070project\holedet\oak | Out-Null
tar -xzf E:\4070project\mmyolo\baidupan\oak_datasetv2.tar.gz -C E:\4070project\holedet\oak
```

解压后请确认配置中的 `data_root` 指向实际数据集路径。

## 重新打包

在 `mmyolo` 根目录执行：

```powershell
.\scripts\pack_baidupan_assets.ps1
```

脚本使用系统自带的 `tar`（bsdtar），对大目录压缩耗时较长，请耐心等待。
