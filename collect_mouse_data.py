import math
import random
import tkinter as tk
import matplotlib.pyplot as plt
import csv
import time
from tkinter import Label

# 创建窗口
root = tk.Tk()
root.attributes('-fullscreen', True)  # 全屏显示

label_n = Label(root, text="n: 0", font=("Helvetica", 16))
label_n.pack()

csv_file_path = "mouse_data.csv"

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# 设置小球的初始位置
ball1_pos = (screen_width/2, screen_height/2)
ball2_pos = (ball1_pos[0] + random.randint(-200, 200),ball1_pos[1] + random.randint(-200, 200))

# 设置小球的半径
ball_radius = 20

# 设置鼠标记录状态
recording = False
mouse_path = []  # 存储 (x, y, timestamp)
start_time = None

n=0

# 鼠标移动事件处理函数
def motion(event):
    global recording, mouse_path, n, start_time
    if recording:
        current_time = time.time()
        if start_time is None:
            start_time = current_time
        mouse_path.append((event.x, event.y, current_time - start_time))

# 鼠标点击事件处理函数
def mouse_click(event):
    global recording, mouse_path, ball2_pos, n, start_time

    if event.x >= ball1_pos[0] - ball_radius and event.x <= ball1_pos[0] + ball_radius and event.y >= ball1_pos[1] - ball_radius and event.y <= ball1_pos[1] + ball_radius:
        recording = True
        start_time = time.time()
        mouse_path = [(event.x, event.y, 0.0)]
    elif event.x >= ball2_pos[0] - ball_radius and event.x <= ball2_pos[0] + ball_radius and event.y >= ball2_pos[1] - ball_radius and event.y <= ball2_pos[1] + ball_radius:
        recording = False
        canvas.delete("ball2")
        #visualize_path(mouse_path)  # 可视化鼠标轨迹
        if len(mouse_path) > 1:
            save_to_csv(mouse_path)
            n = n+1
            if n == 100:
                root.destroy()
            label_n.config(text=f"n: {n}")
        mouse_path = []
        start_time = None

        # 重新生成第二个小球的位置
        ball2_pos = (ball1_pos[0] + random.randint(-200, 200), ball1_pos[1] + random.randint(-200, 200))
        
        # 绘制新的第二个小球
        canvas.create_oval(ball2_pos[0]-ball_radius, ball2_pos[1]-ball_radius, ball2_pos[0]+ball_radius, ball2_pos[1]+ball_radius, fill="blue", tags="ball2")


# 键盘事件处理函数
def key(event):
    if event.keysym == "Escape":
        root.destroy()

def calculate_features(path):
    """计算鼠标轨迹的特征：速度、加速度、角度等"""
    if len(path) < 2:
        return []
    
    features = []
    for i in range(len(path)):
        x, y, t = path[i]
        if i == 0:
            vx, vy = 0, 0
            ax, ay = 0, 0
            angle = 0
            speed = 0
        else:
            prev_x, prev_y, prev_t = path[i-1]
            dt = max(t - prev_t, 0.001)  # 避免除零
            
            # 速度 (像素/秒)
            vx = (x - prev_x) / dt
            vy = (y - prev_y) / dt
            speed = math.sqrt(vx**2 + vy**2)
            
            # 角度 (弧度)
            angle = math.atan2(vy, vx) if (vx != 0 or vy != 0) else 0
            
            # 加速度
            if i > 1:
                prev_prev_x, prev_prev_y, prev_prev_t = path[i-2]
                prev_dt = max(prev_t - prev_prev_t, 0.001)
                prev_vx = (prev_x - prev_prev_x) / prev_dt
                prev_vy = (prev_y - prev_prev_y) / prev_dt
                ax = (vx - prev_vx) / dt
                ay = (vy - prev_vy) / dt
            else:
                ax, ay = 0, 0
        
        features.append({
            'x': x, 'y': y, 't': t,
            'vx': vx, 'vy': vy, 'speed': speed,
            'ax': ax, 'ay': ay,
            'angle': angle
        })
    
    return features

def save_to_csv(path):
    """保存增强的鼠标数据到CSV"""
    if len(path) < 2:
        return
    
    # 计算特征
    features = calculate_features(path)
    
    # 将路径坐标转换为相对于起点的坐标
    start_x, start_y, start_t = path[0]
    x_rel = [f['x'] - start_x for f in features]
    y_rel = [-(f['y'] - start_y) for f in features]  # Y轴翻转
    
    # 提取速度、加速度、角度特征
    speeds = [f['speed'] for f in features]
    angles = [f['angle'] for f in features]
    times = [f['t'] for f in features]
    
    # 选择10个关键点（均匀采样）
    num_points = min(10, len(path))
    if len(path) >= num_points:
        key_indices = [int(i * (len(path) - 1) / (num_points - 1)) for i in range(num_points)]
    else:
        key_indices = list(range(len(path)))
    
    # 构建数据行：目标点 + 10个关键点的完整特征
    row_data = []
    
    # 目标点（最后一个点）
    target_idx = key_indices[-1] if key_indices else -1
    row_data.append(f"{x_rel[target_idx]},{y_rel[target_idx]},{speeds[target_idx]:.2f},{angles[target_idx]:.3f},{times[target_idx]:.3f}")
    
    # 10个关键点的特征（补齐到10个）
    for i in range(10):
        if i < len(key_indices):
            idx = key_indices[i]
            row_data.append(f"{x_rel[idx]},{y_rel[idx]},{speeds[idx]:.2f},{angles[idx]:.3f},{times[idx]:.3f}")
        else:
            # 如果路径太短，用最后一个点填充
            row_data.append(f"{x_rel[-1]},{y_rel[-1]},{speeds[-1]:.2f},{angles[-1]:.3f},{times[-1]:.3f}")
    
    # 打开 CSV 文件进行写操作
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(row_data)


def visualize_path(path):
    # 将路径坐标转换为相对于起点的坐标
    x_rel = [px - path[0][0] for px, py in path]
    y_rel = [-(py - path[0][1]) for px, py in path]

    # 计算每个点相对于起点的距离，用于z轴表示
    distances = [math.sqrt((x_rel[0] - px)**2 + (y_rel[0] - py)**2) for px, py in zip(x_rel, y_rel)]

    # 选择10个关键点
    key_points_indices = [int(i) for i in range(0, len(path), max(1, len(path)//10))]
    key_points_x = [x_rel[i] for i in key_points_indices]
    key_points_y = [y_rel[i] for i in key_points_indices]
    key_points_distances = [distances[i] for i in key_points_indices]

    # 使用z轴信息，通过颜色表示距离的远近
    plt.scatter(key_points_x, key_points_y, c=key_points_distances, cmap='viridis', marker='o', s=50)

    # 在关键点位置添加文本标签，显示终点到起点的距离
    
    plt.text(key_points_x[-1], key_points_y[-1], f'Distance to Origin: {key_points_distances[-1]:.2f}', ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.5))

    # 添加颜色条，表示z轴信息
    plt.colorbar(label='Distance to Endpoint')

    plt.show()

# 绘制小球
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack()
canvas.create_oval(ball1_pos[0]-ball_radius, ball1_pos[1]-ball_radius, ball1_pos[0]+ball_radius, ball1_pos[1]+ball_radius, fill="red")
canvas.create_oval(ball2_pos[0]-ball_radius, ball2_pos[1]-ball_radius, ball2_pos[0]+ball_radius, ball2_pos[1]+ball_radius, fill="blue", tags="ball2")

# 绑定鼠标事件
canvas.bind('<Motion>', motion)
canvas.bind('<Button-1>', mouse_click)

# 绑定键盘事件
root.bind('<Key>', key)

# 运行窗口
root.mainloop()
