import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.model_selection import train_test_split

# 可选导入：ONNX相关（用于模型导出）
try:
    import onnx
    from onnxsim import simplify
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("警告: ONNX相关包未安装，将跳过ONNX模型导出功能")
    print("   如需导出ONNX模型，请运行: pip install onnx onnxsim")

# 读取CSV文件
train_csv_path = 'mouse_data_augmented.csv'
test_csv_path = 'mouse_data_test.csv'

def parse_enhanced_data(csv_path):
    """解析增强的数据格式：x,y,speed,angle,time"""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None, None
    
    df = pd.read_csv(csv_path, header=None)
    if len(df) == 0:
        return None, None
    
    # 解析目标点（第一列）
    targets = []
    for idx, row in df.iterrows():
        target_str = str(row.iloc[0])
        parts = target_str.split(',')
        if len(parts) >= 2:
            targets.append([float(parts[0]), float(parts[1])])
        else:
            continue
    
    # 解析10个关键点（第2-11列）
    key_points = []
    for idx, row in df.iterrows():
        points = []
        for col_idx in range(1, 11):
            if col_idx < len(row):
                point_str = str(row.iloc[col_idx])
                parts = point_str.split(',')
                if len(parts) >= 2:
                    # 提取x, y坐标（忽略speed, angle, time）
                    points.append([float(parts[0]), float(parts[1])])
                else:
                    points.append([0.0, 0.0])
            else:
                points.append([0.0, 0.0])
        # 补齐到10个点
        while len(points) < 10:
            points.append([0.0, 0.0])
        key_points.append(points[:10])
    
    return np.array(targets), np.array(key_points)

def parse_legacy_data(csv_path):
    """解析旧的数据格式（向后兼容）"""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None, None
    
    df = pd.read_csv(csv_path, header=None)
    if len(df) == 0:
        return None, None
    
    # 旧格式：第一列是目标点，后面10列是关键点
    targets = []
    key_points = []
    
    for idx, row in df.iterrows():
        # 解析目标点
        target_str = str(row.iloc[0])
        parts = target_str.split(',')
        if len(parts) >= 2:
            targets.append([float(parts[0]), float(parts[1])])
        else:
            continue
        
        # 解析关键点
        points = []
        for col_idx in range(1, 11):
            if col_idx < len(row):
                point_str = str(row.iloc[col_idx])
                parts = point_str.split(',')
                if len(parts) >= 2:
                    points.append([float(parts[0]), float(parts[1])])
                else:
                    points.append([0.0, 0.0])
            else:
                points.append([0.0, 0.0])
        
        while len(points) < 10:
            points.append([0.0, 0.0])
        key_points.append(points[:10])
    
    return np.array(targets), np.array(key_points)

# 尝试读取数据（优先使用增强格式）
train_targets, train_key_points = parse_enhanced_data(train_csv_path)
if train_targets is None:
    train_targets, train_key_points = parse_legacy_data(train_csv_path)

test_targets, test_key_points = parse_legacy_data(test_csv_path)
if test_targets is None:
    test_targets, test_key_points = parse_enhanced_data(test_csv_path)

if train_targets is None or len(train_targets) == 0:
    print("警告：训练数据为空！请先运行 collect_mouse_data.py 收集数据。")
    exit(1)

# 如果测试数据为空，从训练数据中分割
if test_targets is None or len(test_targets) == 0:
    if len(train_targets) > 10:
        train_targets, test_targets, train_key_points, test_key_points = train_test_split(
            train_targets, train_key_points, test_size=0.2, random_state=42
        )
    else:
        test_targets = train_targets
        test_key_points = train_key_points

print(f"训练样本数: {len(train_targets)}")
print(f"测试样本数: {len(test_targets)}")

# 转换为PyTorch Tensor
# 输入：目标点坐标 (batch_size, 2)
# 输出：10个关键点坐标 (batch_size, 10, 2)
train_inputs = torch.FloatTensor(train_targets)
train_labels = torch.FloatTensor(train_key_points)
test_inputs = torch.FloatTensor(test_targets)
test_labels = torch.FloatTensor(test_key_points)

# 创建自定义Dataset
class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# 创建Dataset和DataLoader
train_dataset = CustomDataset(train_inputs, train_labels)
test_dataset = CustomDataset(test_inputs, test_labels)
batch_size = min(64, len(train_inputs))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 优化的神经网络模型：使用LSTM处理时序信息
class MouseTrajectoryNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, output_points=10):
        super(MouseTrajectoryNet, self).__init__()
        
        # 输入层：将目标点坐标映射到更高维度
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM层：处理时序信息
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 输出层：生成10个点的坐标
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_points * 2)  # 10个点，每个点2个坐标
        )
        
    def forward(self, x):
        # x shape: (batch_size, 2)
        batch_size = x.size(0)
        
        # 输入映射
        x = self.input_fc(x)  # (batch_size, hidden_dim)
        
        # 为LSTM准备输入：将单个点重复10次作为序列
        x_seq = x.unsqueeze(1).repeat(1, 10, 1)  # (batch_size, 10, hidden_dim)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x_seq)  # (batch_size, 10, hidden_dim)
        
        # 只使用最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # 生成输出
        output = self.output_fc(lstm_out)  # (batch_size, 20)
        output = output.view(batch_size, 10, 2)  # (batch_size, 10, 2)
        
        return output

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = MouseTrajectoryNet(input_dim=2, hidden_dim=128, num_layers=2, output_points=10).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

# 训练模型
epochs = 1000
best_test_loss = float('inf')
patience = 50
patience_counter = 0
current_lr = optimizer.param_groups[0]['lr']

print("\n开始训练...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_batches = 0
    
    for batch_inputs, batch_labels in train_dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        output = model(batch_inputs)
        loss = criterion(output, batch_labels)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        train_batches += 1
    
    avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
    
    # 验证
    model.eval()
    test_loss = 0.0
    test_batches = 0
    
    with torch.no_grad():
        for batch_inputs, batch_labels in test_dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            
            output = model(batch_inputs)
            loss = criterion(output, batch_labels)
            test_loss += loss.item()
            test_batches += 1
    
    avg_test_loss = test_loss / test_batches if test_batches > 0 else 0
    old_lr = current_lr
    scheduler.step(avg_test_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # 打印进度
    if (epoch + 1) % 10 == 0 or epoch == 0:
        lr_info = f", LR: {current_lr:.6f}" if old_lr != current_lr else f", LR: {current_lr:.6f}"
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}{lr_info}')
    
    # 学习率变化提示
    if old_lr != current_lr:
        print(f'  → 学习率已调整: {old_lr:.6f} → {current_lr:.6f}')
    
    # 早停机制
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n早停触发！最佳测试损失: {best_test_loss:.6f}")
            # 加载最佳模型
            model.load_state_dict(torch.load('best_model.pth'))
            break

print('\n训练完成！')

# 最终测试
model.eval()
final_test_loss = 0.0
test_batches = 0

print("\n最终测试结果:")
with torch.no_grad():
    for batch_inputs, batch_labels in test_dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        
        output = model(batch_inputs)
        loss = criterion(output, batch_labels)
        final_test_loss += loss.item()
        test_batches += 1
        
        # 显示第一个样本的预测结果
        if test_batches == 1:
            print(f"\n示例输入 (目标点): {batch_inputs[0].cpu().numpy()}")
            print(f"真实轨迹 (10个关键点):")
            print(batch_labels[0].cpu().numpy())
            print(f"预测轨迹 (10个关键点):")
            print(output[0].cpu().numpy())

avg_final_loss = final_test_loss / test_batches if test_batches > 0 else 0
print(f"\n最终测试损失: {avg_final_loss:.6f}")

# 导出ONNX模型（如果可用）
if ONNX_AVAILABLE:
    model.eval()
    onnx_name = 'mouse_model.onnx'
    dummy_input = torch.randn(1, 2).to(device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_name,
            verbose=False,
            input_names=['target_point'],
            output_names=['trajectory_points'],
            dynamic_axes={
                'target_point': {0: 'batch_size'},
                'trajectory_points': {0: 'batch_size'}
            }
        )
        
        # 简化ONNX模型
        try:
            onnx_model = onnx.load(onnx_name)
            simplified_model, check = simplify(onnx_model)
            if check:
                onnx.save(simplified_model, onnx_name)
                print(f"\nONNX模型已导出并简化: {onnx_name}")
            else:
                print(f"\nONNX模型已导出: {onnx_name} (简化失败)")
        except Exception as e:
            print(f"\nONNX模型已导出: {onnx_name} (简化时出错: {e})")
            
    except Exception as e:
        print(f"\n导出ONNX模型时出错: {e}")
else:
    print("\n跳过ONNX模型导出（相关包未安装）")

print("\n完成！")
