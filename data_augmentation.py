"""
数据增强和预处理工具
用于扩充训练数据集，提高模型泛化能力
"""

import pandas as pd
import numpy as np
import csv
import random
import math

def augment_trajectory(x_coords, y_coords, methods=['noise', 'scale', 'rotate']):
    """
    对单个轨迹进行数据增强
    
    参数:
        x_coords: x坐标列表
        y_coords: y坐标列表
        methods: 增强方法列表
    
    返回:
        增强后的x, y坐标列表
    """
    x = np.array(x_coords)
    y = np.array(y_coords)
    
    # 添加噪声
    if 'noise' in methods:
        noise_scale = 0.05  # 噪声比例
        x_range = np.max(x) - np.min(x) if len(x) > 0 else 1
        y_range = np.max(y) - np.min(y) if len(y) > 0 else 1
        x += np.random.normal(0, abs(x_range) * noise_scale, len(x))
        y += np.random.normal(0, abs(y_range) * noise_scale, len(y))
    
    # 缩放
    if 'scale' in methods:
        scale_factor = random.uniform(0.8, 1.2)
        x *= scale_factor
        y *= scale_factor
    
    # 旋转
    if 'rotate' in methods:
        angle = random.uniform(-15, 15) * math.pi / 180  # 随机旋转-15到15度
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        x, y = x_new, y_new
    
    return x.tolist(), y.tolist()

def augment_csv_file(input_file, output_file, augmentation_factor=3):
    """
    对CSV文件中的数据进行增强
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        augmentation_factor: 每个样本增强的次数
    """
    if not pd.io.common.file_exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        return
    
    df = pd.read_csv(input_file, header=None)
    if len(df) == 0:
        print(f"警告: 文件 {input_file} 为空")
        return
    
    augmented_data = []
    
    for idx, row in df.iterrows():
        # 解析目标点
        target_str = str(row.iloc[0])
        target_parts = target_str.split(',')
        if len(target_parts) < 2:
            continue
        
        target_x = float(target_parts[0])
        target_y = float(target_parts[1])
        
        # 解析关键点
        key_points_x = []
        key_points_y = []
        
        for col_idx in range(1, 11):
            if col_idx < len(row):
                point_str = str(row.iloc[col_idx])
                parts = point_str.split(',')
                if len(parts) >= 2:
                    key_points_x.append(float(parts[0]))
                    key_points_y.append(float(parts[1]))
                else:
                    key_points_x.append(0.0)
                    key_points_y.append(0.0)
            else:
                key_points_x.append(0.0)
                key_points_y.append(0.0)
        
        # 补齐到10个点
        while len(key_points_x) < 10:
            key_points_x.append(0.0)
            key_points_y.append(0.0)
        
        # 原始数据
        original_row = [f"{target_x},{target_y}"] + [
            f"{key_points_x[i]},{key_points_y[i]}" for i in range(10)
        ]
        augmented_data.append(original_row)
        
        # 增强数据
        for _ in range(augmentation_factor):
            aug_x, aug_y = augment_trajectory(key_points_x, key_points_y)
            aug_target_x, aug_target_y = augment_trajectory([target_x], [target_y])
            
            augmented_row = [f"{aug_target_x[0]},{aug_target_y[0]}"] + [
                f"{aug_x[i]},{aug_y[i]}" for i in range(10)
            ]
            augmented_data.append(augmented_row)
    
    # 保存增强后的数据
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in augmented_data:
            csv_writer.writerow(row)
    
    print(f"✓ 数据增强完成:")
    print(f"  原始样本数: {len(df)}")
    print(f"  增强后样本数: {len(augmented_data)}")
    print(f"  增强倍数: {augmentation_factor + 1}")
    print(f"  输出文件: {output_file}")

def normalize_data(input_file, output_file):
    """
    数据归一化：将坐标归一化到[-1, 1]范围
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
    """
    if not pd.io.common.file_exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        return
    
    df = pd.read_csv(input_file, header=None)
    if len(df) == 0:
        print(f"警告: 文件 {input_file} 为空")
        return
    
    all_x = []
    all_y = []
    
    # 收集所有坐标值
    for idx, row in df.iterrows():
        target_str = str(row.iloc[0])
        target_parts = target_str.split(',')
        if len(target_parts) >= 2:
            all_x.append(float(target_parts[0]))
            all_y.append(float(target_parts[1]))
        
        for col_idx in range(1, 11):
            if col_idx < len(row):
                point_str = str(row.iloc[col_idx])
                parts = point_str.split(',')
                if len(parts) >= 2:
                    all_x.append(float(parts[0]))
                    all_y.append(float(parts[1]))
    
    # 计算归一化参数
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    
    print(f"数据范围:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
    
    # 归一化数据
    normalized_data = []
    for idx, row in df.iterrows():
        target_str = str(row.iloc[0])
        target_parts = target_str.split(',')
        if len(target_parts) < 2:
            continue
        
        # 归一化目标点
        norm_target_x = 2 * (float(target_parts[0]) - x_min) / x_range - 1
        norm_target_y = 2 * (float(target_parts[1]) - y_min) / y_range - 1
        
        normalized_row = [f"{norm_target_x},{norm_target_y}"]
        
        # 归一化关键点
        for col_idx in range(1, 11):
            if col_idx < len(row):
                point_str = str(row.iloc[col_idx])
                parts = point_str.split(',')
                if len(parts) >= 2:
                    norm_x = 2 * (float(parts[0]) - x_min) / x_range - 1
                    norm_y = 2 * (float(parts[1]) - y_min) / y_range - 1
                    normalized_row.append(f"{norm_x},{norm_y}")
                else:
                    normalized_row.append("0,0")
            else:
                normalized_row.append("0,0")
        
        normalized_data.append(normalized_row)
    
    # 保存归一化后的数据
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in normalized_data:
            csv_writer.writerow(row)
    
    print(f"✓ 数据归一化完成:")
    print(f"  样本数: {len(normalized_data)}")
    print(f"  输出文件: {output_file}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  数据增强: python data_augmentation.py augment <输入文件> <输出文件> [增强倍数]")
        print("  数据归一化: python data_augmentation.py normalize <输入文件> <输出文件>")
        print("\n示例:")
        print("  python data_augmentation.py augment mouse_data.csv mouse_data_augmented.csv 3")
        print("  python data_augmentation.py normalize mouse_data.csv mouse_data_normalized.csv")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'augment':
        if len(sys.argv) < 4:
            print("错误: 缺少参数")
            print("用法: python data_augmentation.py augment <输入文件> <输出文件> [增强倍数]")
            sys.exit(1)
        
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        factor = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        
        augment_csv_file(input_file, output_file, factor)
    
    elif command == 'normalize':
        if len(sys.argv) < 4:
            print("错误: 缺少参数")
            print("用法: python data_augmentation.py normalize <输入文件> <输出文件>")
            sys.exit(1)
        
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        
        normalize_data(input_file, output_file)
    
    else:
        print(f"错误: 未知命令 '{command}'")
        sys.exit(1)

