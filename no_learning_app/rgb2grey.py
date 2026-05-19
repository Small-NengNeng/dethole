#读取目录；/home/xyzt2204/project/holedet/oak/no_learning_method/img
#转化为grey 灰度图像 保存到/home/xyzt2204/project/holedet/oak/no_learning_method/grey

import os
import cv2
from glob import glob


def convert_rgb_to_grey(input_dir, output_dir):
    """
    读取指定目录下的图片文件，转换为灰度图像并保存
    
    Args:
        input_dir: 输入图片目录路径
        output_dir: 输出灰度图片目录路径
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 获取目录下所有图片文件（支持jpg, jpeg, png等格式）
    image_files = (glob(os.path.join(input_dir, '*.jpg')) + 
                   glob(os.path.join(input_dir, '*.JPG')) +
                   glob(os.path.join(input_dir, '*.jpeg')) +
                   glob(os.path.join(input_dir, '*.JPEG')) +
                   glob(os.path.join(input_dir, '*.png')) +
                   glob(os.path.join(input_dir, '*.PNG')))
    
    if not image_files:
        print(f"在目录 {input_dir} 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 处理每张图片
    for img_path in image_files:
        # 读取图片
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        
        # 转换为灰度图
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 生成输出文件路径（保持原文件名）
        filename = os.path.basename(img_path)
        # 如果原文件不是jpg格式，统一保存为jpg
        name_without_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name_without_ext}.jpg")
        
        # 保存灰度图像
        cv2.imwrite(output_path, img_grey)
        print(f"已转换并保存: {filename} -> {os.path.basename(output_path)}")
    
    print(f"\n转换完成！共处理 {len(image_files)} 张图片")


if __name__ == "__main__":
    # 输入目录
    input_directory = "/home/xyzt2204/project/holedet/oak/no_learning_method/img"
    # 输出目录
    output_directory = "/home/xyzt2204/project/holedet/oak/no_learning_method/grey"
    
    if not os.path.isdir(input_directory):
        print(f"错误: 输入目录 {input_directory} 不存在")
    else:
        convert_rgb_to_grey(input_directory, output_directory)