# 百度网盘大文件说明

GitHub 仓库仅包含代码与配置，下列资源请从百度网盘下载。

## 数据集

| 项目 | 说明 |
|------|------|
| 文件名 | `real_trainval_yolo.7z` |
| 链接 | https://pan.baidu.com/s/1x8ScM9siq7yv2JvgdL4mqg |
| 提取码 | `8ar2` |

建议保存并解压到：

```text
E:\4070project\holedet\oak\real_trainval_yolo.7z
E:\4070project\holedet\oak\<解压目录>\
```

解压后请在训练配置中设置正确的 `data_root`。

## 训练权重与实验输出

| 项目 | 说明 |
|------|------|
| 文件名 | `mmyolo_work_dirs.tar.gz` |
| 链接 | https://pan.baidu.com/s/1VIYYhSidqYELLCOrKkracg |
| 提取码 | `ue4p` |
| 内容 | 训练 checkpoint（`.pth`）、日志、可视化等 |

建议下载到 `E:\4070project\mmyolo\baidupan\` 后解压。

解压示例：

```powershell
tar -xzf E:\4070project\mmyolo\baidupan\mmyolo_work_dirs.tar.gz -C E:\4070project\mmyolo\mmyolo
```

## 重新打包权重（可选）

仅打包 `work_dirs` 时，在 `mmyolo` 根目录执行：

```powershell
.\scripts\pack_baidupan_assets.ps1
```
