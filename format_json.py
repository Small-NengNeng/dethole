import json
import os

def format_json_file(input_file, output_file=None):
    """
    格式化 JSON 文件
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为 None 则覆盖原文件
    """
    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果未指定输出文件，则覆盖原文件
    if output_file is None:
        output_file = input_file
    
    # 写入格式化后的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    # 设置输入文件路径
    input_file = '/home/xyzt2204/project/holedet/camera/annotations/instances_train2017.json'
    
    # 格式化 JSON 文件
    format_json_file(input_file)
    print(f'JSON 文件已格式化完成：{input_file}') 