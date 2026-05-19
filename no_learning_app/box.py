#实现一个函数 输入一个图片 给出bbox list x y w h 
#测试main函数 路径为/home/xyzt2204/project/holedet/oak/no_learning_method/img/1.jpg 可视化bbox结果

import cv2
import numpy as np
from typing import List, Tuple, Union


class BBoxDetector:
    """
    目标检测类，用于从RGB图像中提取目标bbox信息
    """
    
    def __init__(self, gray_min=0, gray_max=50, 
                 min_size=60, min_area=220, 
                 aspect_ratio_range=(0.5, 1.8),
                 circularity_threshold=0.01,
                 fill_ratio_threshold=0.3):
        """
        初始化检测器参数
        
        Args:
            gray_min: 灰度区间最小值，默认0
            gray_max: 灰度区间最大值，默认50
            min_size: 最小宽度/高度，默认60
            min_area: 最小面积，默认220
            aspect_ratio_range: 宽高比范围 (min, max)，默认(0.5, 1.8)
            circularity_threshold: 圆形度阈值，默认0.01
            fill_ratio_threshold: 填充比阈值，默认0.3
        """
        self.gray_min = gray_min
        self.gray_max = gray_max
        self.min_size = min_size
        self.min_area = min_area
        self.aspect_ratio_min, self.aspect_ratio_max = aspect_ratio_range
        self.circularity_threshold = circularity_threshold
        self.fill_ratio_threshold = fill_ratio_threshold
    
    def rgb_to_gray(self, img: Union[str, np.ndarray]) -> np.ndarray:
        """
        将RGB图像转换为灰度图像
        
        Args:
            img: 图像路径或numpy数组（BGR格式）
            
        Returns:
            灰度图像数组
        """
        if isinstance(img, str):
            # 如果是路径，读取图像
            img_bgr = cv2.imread(img)
            if img_bgr is None:
                raise ValueError(f"无法读取图片: {img}")
        else:
            # 如果已经是numpy数组，直接使用
            img_bgr = img
        
        # 转换为灰度图
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return img_gray
    
    def create_binary_mask(self, img_gray: np.ndarray) -> np.ndarray:
        """
        根据灰度区间创建二值图
        
        Args:
            img_gray: 灰度图像
            
        Returns:
            二值图（0或255）
        """
        binary_mask = np.zeros_like(img_gray)
        binary_mask[(img_gray >= self.gray_min) & (img_gray <= self.gray_max)] = 255
        return binary_mask
    
    def filter_bbox(self, x: int, y: int, w: int, h: int, area: int, 
                   labels: np.ndarray, label_id: int, 
                   img_height: int, img_width: int) -> bool:
        """
        判断bbox是否通过过滤条件
        
        Args:
            x, y, w, h: bbox坐标和尺寸
            area: 连通域面积
            labels: 连通域标签图
            label_id: 当前连通域标签ID
            img_height, img_width: 图像尺寸
            
        Returns:
            True表示通过过滤，False表示被过滤
        """
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
        
        # 过滤条件1：宽度或高度小于阈值
        if w < self.min_size or h < self.min_size:
            return False
        
        # 过滤条件2：像素数量（面积）少于阈值
        if area < self.min_area:
            return False
        
        # 过滤条件3：宽高比不在范围内（如果矩形框紧贴边界，则跳过此判断）
        if not touches_boundary:
            if h > 0:  # 避免除零
                aspect_ratio = w / h
                if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:
                    return False
            else:
                return False
        
        # 过滤条件4：形状过滤：检测是否为球形（圆形）
        # 提取当前连通域的mask
        component_mask = (labels == label_id).astype(np.uint8) * 255
        
        # 找到轮廓
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return False
        
        # 使用最大轮廓（通常只有一个）
        contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            # 计算圆形度：circularity = 4π * area / perimeter²
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 圆形度阈值检查
            if circularity < self.circularity_threshold:
                return False
            
            # 使用最小外接圆进一步验证
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            min_circle_area = np.pi * radius * radius
            
            # 实际面积与最小外接圆面积的比值
            if min_circle_area > 0:
                fill_ratio = area / min_circle_area
                # 填充比检查
                if fill_ratio < self.fill_ratio_threshold:
                    return False
        else:
            return False
        
        return True
    
    def detect(self, img: Union[str, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        从RGB图像中检测目标并返回bbox列表
        
        Args:
            img: 图像路径或numpy数组（BGR格式）
            
        Returns:
            bbox列表，每个元素为(x, y, w, h)元组
        """
        # 1. RGB转灰度
        img_gray = self.rgb_to_gray(img)
        
        # 2. 创建二值图
        binary_mask = self.create_binary_mask(img_gray)
        
        # 3. 使用连通域分析计算矩形框
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=4)
        
        # 获取图像尺寸
        img_height, img_width = img_gray.shape
        
        # 4. 遍历所有连通域，过滤并收集bbox
        bbox_list = []
        for i in range(1, num_labels):  # 从1开始，跳过背景
            # stats包含: [x, y, width, height, area]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 应用过滤条件
            if self.filter_bbox(x, y, w, h, area, labels, i, img_height, img_width):
                bbox_list.append((x, y, w, h))
        
        return bbox_list
    
    def visualize(self, img: Union[str, np.ndarray], 
                  bbox_list: List[Tuple[int, int, int, int]],
                  output_path: str = None) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            img: 原始图像路径或numpy数组
            bbox_list: bbox列表
            output_path: 保存路径（可选）
            
        Returns:
            可视化图像（BGR格式）
        """
        # 读取原始图像
        if isinstance(img, str):
            vis_img = cv2.imread(img)
            if vis_img is None:
                raise ValueError(f"无法读取图片: {img}")
        else:
            vis_img = img.copy()
        
        # 绘制bbox
        for x, y, w, h in bbox_list:
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 保存结果（如果指定了路径）
        if output_path:
            cv2.imwrite(output_path, vis_img)
            print(f"已保存可视化结果到: {output_path}")
        
        return vis_img


if __name__ == "__main__":
    # 测试图片路径
    test_image_path = "/home/xyzt2204/project/holedet/oak/no_learning_method/img/1.jpg"
    
    # 创建检测器实例（使用默认参数，可根据需要调整）
    detector = BBoxDetector(gray_min=0, gray_max=50)
    
    # 检测bbox
    print(f"正在处理图片: {test_image_path}")
    bbox_list = detector.detect(test_image_path)
    
    print(f"检测到 {len(bbox_list)} 个目标")
    for i, (x, y, w, h) in enumerate(bbox_list, 1):
        print(f"  BBox {i}: x={x}, y={y}, w={w}, h={h}")
    
    # 可视化结果
    vis_img = detector.visualize(test_image_path, bbox_list)
    
    # 显示结果（如果环境支持）
    try:
        cv2.imshow("Detection Result", vis_img)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("无法显示图像窗口，但可视化图像已生成")
    
    # 可选：保存可视化结果
    output_path = "/home/xyzt2204/project/holedet/oak/no_learning_method/box_result.jpg"
    detector.visualize(test_image_path, bbox_list, output_path)
