#使用box 下的BBoxDetector 类进行detect 获取bbox 保存bbox list[xywh]为json 名称与图像名称一致 并可视化bbox 可视化名称与其一致
#批量处理指定目录下的图片/home/xyzt2204/project/holedet/oak/real_yolo/images 格式为jpg 输出结果到/home/xyzt2204/project/holedet/oak/no_learning_method/result/运行时间

import os
import json
import cv2
import numpy as np
from datetime import datetime
from glob import glob
from box import BBoxDetector


def batch_process_images(input_dir, output_base_dir, gray_min=0, gray_max=50):
    """
    批量处理指定目录下的JPG图片，使用BBoxDetector进行检测，保存bbox为JSON并可视化
    
    Args:
        input_dir: 输入图片目录路径
        output_base_dir: 输出结果基础目录路径
        gray_min: 灰度区间最小值，默认0
        gray_max: 灰度区间最大值，默认50
    """
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, timestamp)
    
    # 创建输出子目录
    json_dir = os.path.join(output_dir, "json")
    vis_dir = os.path.join(output_dir, "visualization")
    
    for dir_path in [output_dir, json_dir, vis_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建输出目录: {dir_path}")
    
    # 获取目录下所有jpg文件（不区分大小写）
    # 注意：目录下不一定只有图片，需要过滤出.jpg后缀的文件
    jpg_files = (glob(os.path.join(input_dir, '*.jpg')) + 
                 glob(os.path.join(input_dir, '*.JPG')) +
                 glob(os.path.join(input_dir, '*.jpeg')) +
                 glob(os.path.join(input_dir, '*.JPEG')) + glob(os.path.join(input_dir, '*.png')))
    
    if not jpg_files:
        print(f"在目录 {input_dir} 中未找到JPG图片文件")
        return
    
    print(f"找到 {len(jpg_files)} 个JPG文件")
    print(f"输出目录: {output_dir}")
    
    # 创建检测器实例
    detector = BBoxDetector(gray_min=gray_min, gray_max=gray_max)
    
    # 处理每张图片
    success_count = 0
    for img_path in jpg_files[0:40]:
        try:
            # 获取文件名（不含扩展名）
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            print(f"\n正在处理: {filename}")
            
            # 使用BBoxDetector进行检测
            bbox_list = detector.detect(img_path)
            
            print(f"检测到 {len(bbox_list)} 个目标")
            
            # 将bbox列表转换为可序列化的格式（元组转列表，numpy类型转Python原生类型）
            bbox_list_serializable = [[int(x), int(y), int(w), int(h)] for x, y, w, h in bbox_list]
            
            # 立即保存bbox为JSON文件（处理一张图保存一个结果）
            json_path = os.path.join(json_dir, f"{name_without_ext}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(bbox_list_serializable, f, indent=2, ensure_ascii=False)
            print(f"已保存JSON: {os.path.basename(json_path)}")
            
            # 立即可视化bbox并保存（处理一张图保存一个结果）
            vis_path = os.path.join(vis_dir, f"{name_without_ext}.jpg")
            detector.visualize(img_path, bbox_list, vis_path)
            print(f"已保存可视化结果: {os.path.basename(vis_path)}")
            
            success_count += 1
            
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")
            continue
    
    print(f"\n处理完成！成功处理 {success_count}/{len(jpg_files)} 张图片")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    # 输入目录
    input_directory = "/home/xyzt2204/project/holedet/camera/images"
    # 输出基础目录
    output_base_directory = "/home/xyzt2204/project/holedet/oak/no_learning_method/camera_vis"
    
    # 设置灰度区间（可根据需要调整）
    gray_min_value = 0
    gray_max_value = 20
    
    if not os.path.isdir(input_directory):
        print(f"错误: 输入目录 {input_directory} 不存在")
    else:
        batch_process_images(input_directory, output_base_directory, 
                           gray_min_value, gray_max_value)
