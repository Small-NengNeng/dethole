#实现脚本
#1 读取指定目录下的图片文件 后缀为jpg 图片为彩色

#2.转化为灰度

#统计灰度直方图 并使用plot画出

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def process_images_in_directory(directory_path):
    """
    读取指定目录下的JPG图片，转换为灰度，并绘制灰度直方图
    
    Args:
        directory_path: 图片目录路径
    """
    # 获取目录下所有jpg文件
    jpg_files = glob(os.path.join(directory_path, '*.jpg')) + \
                glob(os.path.join(directory_path, '*.JPG'))
    
    if not jpg_files:
        print(f"在目录 {directory_path} 中未找到JPG图片文件")
        return
    
    print(f"找到 {len(jpg_files)} 个JPG文件")
    
    # 处理每张图片
    for img_path in jpg_files:
        # 读取彩色图片
        img_color = cv2.imread(img_path)
        
        if img_color is None:
            print(f"无法读取图片: {img_path}")
            continue
        
        # 转换为灰度图
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # 计算灰度直方图
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        
        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.plot(hist, color='black')
        plt.title(f'灰度直方图 - {os.path.basename(img_path)}')
        plt.xlabel('灰度值')
        plt.ylabel('像素数量')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 256])
        plt.show()
        
        print(f"已处理: {os.path.basename(img_path)}")


if __name__ == "__main__":
    # 指定图片目录路径，可以根据需要修改

    image_directory = "/home/xyzt2204/project/holedet/oak/no_learning_method/img"
    
    if not os.path.isdir(image_directory):
        print(f"错误: 目录 {image_directory} 不存在")
    else:
        process_images_in_directory(image_directory)
