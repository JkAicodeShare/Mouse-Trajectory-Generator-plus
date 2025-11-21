# 鼠标轨迹预测项目 - 使用说明

## 📁 项目文件说明

### 核心文件

#### 1. `collect_mouse_data.py` - 数据收集工具
**作用：** 收集鼠标移动轨迹数据用于训练

**功能：**
- 创建全屏窗口，显示红色起点小球和蓝色终点小球
- 点击红色小球开始记录鼠标轨迹
- 点击蓝色小球停止记录并保存数据
- 自动计算速度、加速度、角度等特征
- 保存为CSV格式（`mouse_data.csv`）

**使用方法：**
```bash
python collect_mouse_data.py
```

**操作说明：**
- 点击红色小球 → 开始记录
- 移动鼠标到蓝色小球 → 自动记录轨迹
- 点击蓝色小球 → 保存数据
- 按ESC键 → 退出程序
- 收集100条数据后自动退出（可在代码中修改）

---

#### 2. `train.py` - 模型训练脚本
**作用：** 训练神经网络模型，学习预测鼠标轨迹

**功能：**
- 读取CSV数据并自动分割训练集/测试集
- 使用LSTM神经网络进行训练
- 自动保存最佳模型（`best_model.pth`）
- 显示训练进度和损失值
- 支持早停和学习率调度
- 导出ONNX模型（可选）

**使用方法：**
```bash
python train.py
```

**输出文件：**
- `best_model.pth` - 训练好的模型权重
- `mouse_model.onnx` - ONNX格式模型（如果安装了ONNX）

**训练参数：**
- 训练轮数：1000 epochs（可早停）
- 批次大小：64
- 学习率：0.001（自适应调整）
- 隐藏层维度：128
- LSTM层数：2

---

#### 3. `test_api.py` - API服务器
**作用：** 提供RESTful API接口，用于调用训练好的模型进行预测

**功能：**
- 加载训练好的模型
- 提供HTTP API接口
- 支持单个预测和批量预测
- 完整的错误处理和文档

**使用方法：**
```bash
python test_api.py
```

**API端点：**
- `GET /` - API文档首页
- `GET /health` - 健康检查
- `POST /predict` - 单个预测
- `POST /predict_batch` - 批量预测

**API地址：** `http://localhost:5000`

**请求示例：**
```python
import requests
response = requests.post(
    "http://localhost:5000/predict",
    json={"target_point": [100, 200]}
)
```

---

#### 4. `test_client.py` - API测试客户端
**作用：** 测试API服务器的功能

**功能：**
- 测试健康检查接口
- 测试单个预测接口
- 测试批量预测接口
- 测试错误处理

**使用方法：**
```bash
# 先启动API服务器（另一个终端）
python test_api.py

# 然后运行测试客户端
python test_client.py
```

---

#### 5. `data_augmentation.py` - 数据增强工具
**作用：** 对训练数据进行增强，扩充数据集

**功能：**
- 添加噪声增强
- 缩放变换（0.8-1.2倍）
- 旋转变换（-15°到15°）
- 数据归一化

**使用方法：**
```bash
# 数据增强（每个样本增强3次）
python data_augmentation.py augment mouse_data.csv mouse_data_augmented.csv 3

# 数据归一化
python data_augmentation.py normalize mouse_data.csv mouse_data_normalized.csv
```

**参数说明：**
- `augment` - 数据增强命令
- `normalize` - 数据归一化命令
- 最后一个数字是增强倍数

---

#### 6. `show.py` - 简单可视化工具
**作用：** 可视化鼠标轨迹数据点

**功能：**
- 使用matplotlib绘制散点图
- 显示轨迹点分布

**使用方法：**
```bash
python show.py
```

**注意：** 这是一个简单的示例脚本，可以修改其中的数据来可视化不同的轨迹。

---

#### 7. `visualize_results.py` - 训练结果可视化（新增）
**作用：** 美观地展示训练结果和模型预测效果

**功能：**
- 可视化训练损失曲线
- 对比真实轨迹和预测轨迹
- 显示多个测试样本的预测效果
- 生成美观的图表

**使用方法：**
```bash
python visualize_results.py
```

---

## 🚀 完整使用流程

### 第一步：收集数据
```bash
python collect_mouse_data.py
```
- 点击红色小球开始
- 移动鼠标到蓝色小球
- 重复100次（或修改代码中的数量）

### 第二步：数据增强（可选）
```bash
python data_augmentation.py augment mouse_data.csv mouse_data_augmented.csv 5
```

### 第三步：训练模型
```bash
python train.py
```
- 等待训练完成
- 查看训练损失和测试损失
- 模型会自动保存为 `best_model.pth`

### 第四步：可视化结果（可选）
```bash
python visualize_results.py
```
- 查看训练曲线
- 查看预测效果对比

### 第五步：启动API服务器
```bash
python test_api.py
```
- 保持终端窗口打开
- API在 `http://localhost:5000` 运行

### 第六步：测试API
```bash
# 在另一个终端
python test_client.py
```

---

## 📊 数据文件说明

### `mouse_data.csv` - 训练数据
- 格式：每行包含1个目标点和10个关键点
- 字段：x, y坐标（可能包含速度、角度等特征）
- 用途：用于训练模型

### `mouse_data_test.csv` - 测试数据
- 格式：与训练数据相同
- 用途：用于评估模型性能
- 如果不存在，会自动从训练数据中分割

### `best_model.pth` - 训练好的模型
- 格式：PyTorch模型权重文件
- 用途：API服务器加载此文件进行预测

---

## 🔧 配置和自定义

### 修改数据收集数量
编辑 `collect_mouse_data.py`：
```python
if n == 100:  # 改为你想要的数字
    root.destroy()
```

### 修改训练参数
编辑 `train.py`：
```python
epochs = 1000  # 训练轮数
batch_size = 64  # 批次大小
hidden_dim = 128  # 隐藏层维度
```

### 修改API端口
编辑 `test_api.py`：
```python
app.run(host='0.0.0.0', port=5000, debug=False)  # 修改端口号
```

---

## ❓ 常见问题

**Q: 训练时提示数据为空？**
A: 先运行 `python collect_mouse_data.py` 收集数据

**Q: API无法启动？**
A: 确保已安装Flask：`pip install flask`

**Q: 模型文件不存在？**
A: 先运行 `python train.py` 训练模型

**Q: 预测结果不准确？**
A: 
- 收集更多训练数据（建议500+条）
- 使用数据增强
- 调整模型超参数

**Q: 端口被占用？**
A: 修改 `test_api.py` 中的端口号

---

## 📝 文件依赖关系

```
collect_mouse_data.py
    ↓ 生成
mouse_data.csv
    ↓ 读取
train.py
    ↓ 生成
best_model.pth
    ↓ 加载
test_api.py
    ↓ 调用
test_client.py
```

---

## 🎯 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **收集数据**
   ```bash
   python collect_mouse_data.py
   ```

3. **训练模型**
   ```bash
   python train.py
   ```

4. **查看结果**
   ```bash
   python visualize_results.py
   ```

5. **启动API**
   ```bash
   python test_api.py
   ```

---



