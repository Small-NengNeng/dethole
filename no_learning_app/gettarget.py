#读取目录/home/xyzt2204/project/holedet/oak/no_learning_method/grey下的图片批处理

#设置灰度区间 min max 将图像转化为二值图 保存到对应位置 /home/xyzt2204/project/holedet/oak/no_learning_method/mask 名字与原图像相同

#使用opencv连通域计算灰度区间的连通域及其矩形框 画出bbox的可视化结果 保存到 /home/xyzt2204/project/holedet/oak/no_learning_method/vis_mask

#若保存路径不存在则创建

import os
import cv2
import numpy as np
from glob import glob


def process_images(input_dir, mask_dir, vis_mask_dir, gray_min=0, gray_max=255):
    """
    读取灰度图片，进行二值化处理，计算连通域并绘制bbox可视化
    
    Args:
        input_dir: 输入灰度图片目录路径
        mask_dir: 输出二值图目录路径
        vis_mask_dir: 输出可视化结果目录路径
        gray_min: 灰度区间最小值，默认0
        gray_max: 灰度区间最大值，默认255
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
        print(f"创建输出目录: {mask_dir}")
    
    if not os.path.exists(vis_mask_dir):
        os.makedirs(vis_mask_dir)
        print(f"创建输出目录: {vis_mask_dir}")
    
    # 获取目录下所有图片文件
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
    print(f"灰度区间: [{gray_min}, {gray_max}]")
    
    # 处理每张图片
    for img_path in image_files:
        # 读取灰度图片
        img_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img_grey is None:
            print(f"无法读取图片: {img_path}")
            continue
        
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # 1. 根据灰度区间创建二值图
        # 创建掩码：在灰度区间内的像素为255，否则为0
        binary_mask = np.zeros_like(img_grey)
        binary_mask[(img_grey >= gray_min) & (img_grey <= gray_max)] = 255
        
        # 保存二值图到mask目录
        mask_output_path = os.path.join(mask_dir, f"{name_without_ext}.jpg")
        cv2.imwrite(mask_output_path, binary_mask)
        print(f"已保存二值图: {filename} -> {os.path.basename(mask_output_path)}")
        
        # 2. 使用连通域分析计算矩形框
        # 使用connectedComponentsWithStats获取连通域信息
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=4)
        
        # 创建可视化图像（将灰度图转为BGR以便绘制彩色框）
        vis_img = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)
        
        # 获取图像尺寸
        img_height, img_width = img_grey.shape
        
        # 遍历所有连通域（跳过背景，label 0）
        bbox_count = 0
        filtered_count = 0
        for i in range(1, num_labels):  # 从1开始，跳过背景
            # stats包含: [x, y, width, height, area]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 计算矩形框的四个角点
            x1, y1 = x, y  # 左上角
            x2, y2 = x + w - 1, y  # 右上角
            x3, y3 = x, y + h - 1  # 左下角
            x4, y4 = x + w - 1, y + h - 1  # 右下角
            
            # 判断是否有角点紧贴图像边界（距离边界小于等于1个像素）
            touches_boundary = (
                x1 <= 1 or y1 <= 1 or  # 左上角
                x2 >= img_width - 2 or y2 <= 1 or  # 右上角
                x3 <= 1 or y3 >= img_height - 2 or  # 左下角
                x4 >= img_width - 2 or y4 >= img_height - 2  # 右下角
            )
            
            # 过滤条件：
            # 1. 宽度或高度小于60的去除
            if w < 60 or h < 60:
                filtered_count += 1
                continue
            
            # 2. 像素数量（面积）少于220的去除
            if area < 220:
                filtered_count += 1
                continue
            
            # 3. 宽高比不在0.5-1.8之间的去除（如果矩形框紧贴边界，则跳过此判断）
            if not touches_boundary:
                if h > 0:  # 避免除零
                    aspect_ratio = w / h
                    if aspect_ratio < 0.5 or aspect_ratio > 1.8:
                        filtered_count += 1
                        continue
                else:
                    filtered_count += 1
                    continue
            
            # 4. 形状过滤：检测是否为球形（圆形）
            # 提取当前连通域的mask
            component_mask = (labels == i).astype(np.uint8) * 255
            
            # 找到轮廓
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                filtered_count += 1
                continue
            
            # 使用最大轮廓（通常只有一个）
            contour = max(contours, key=cv2.contourArea)
            
            # 计算轮廓周长
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                # 计算圆形度：circularity = 4π * area / perimeter²
                # 对于完美圆形，值为1；值越小，形状越不规则
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # 圆形度阈值：小于0.7认为不是球形（可根据实际情况调整）
                if circularity < 0.01:
                    filtered_count += 1
                    continue
                
                # 可选：使用最小外接圆进一步验证
                # 计算最小外接圆
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                min_circle_area = np.pi * radius * radius
                
                # 实际面积与最小外接圆面积的比值，越接近1越圆
                if min_circle_area > 0:
                    fill_ratio = area / min_circle_area
                    # 填充比小于0.7认为不是球形（可根据实际情况调整）
                    if fill_ratio < 0.3:
                        filtered_count += 1
                        continue
            else:
                filtered_count += 1
                continue
            
            # 通过所有过滤条件，绘制矩形框（绿色，线宽2）
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 可选：在框上标注面积信息
            # cv2.putText(vis_img, f'Area:{area}', (x, y-5), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            bbox_count += 1
        
        # 保存可视化结果
        vis_output_path = os.path.join(vis_mask_dir, f"{name_without_ext}.jpg")
        cv2.imwrite(vis_output_path, vis_img)
        print(f"已保存可视化结果: {filename} -> {os.path.basename(vis_output_path)} (找到 {bbox_count} 个有效连通域，过滤 {filtered_count} 个)")
    
    print(f"\n处理完成！共处理 {len(image_files)} 张图片")


if __name__ == "__main__":
    # 输入目录（灰度图片）
    input_directory = "/home/xyzt2204/project/holedet/oak/no_learning_method/grey"
    # 输出目录（二值图）
    mask_directory = "/home/xyzt2204/project/holedet/oak/no_learning_method/mask"
    # 输出目录（可视化结果）
    vis_mask_directory = "/home/xyzt2204/project/holedet/oak/no_learning_method/vis_mask"
    
    # 设置灰度区间（可根据实际需求调整）
    gray_min_value = 0
    gray_max_value = 50
    
    if not os.path.isdir(input_directory):
        print(f"错误: 输入目录 {input_directory} 不存在")
    else:
        process_images(input_directory, mask_directory, vis_mask_directory, 
                      gray_min_value, gray_max_value)
