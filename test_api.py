"""
简单的鼠标轨迹预测API
使用方法：
1. 启动API: python test_api.py
2. 发送POST请求到 http://localhost:5000/predict
   Body格式: {"target_point": [100, 200]}
3. 返回预测的10个关键点坐标
"""

from flask import Flask, request, jsonify
import torch
import numpy as np
import os

app = Flask(__name__)

# 加载模型
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """加载训练好的模型"""
    global model
    
    # 定义模型架构（必须与train.py中的一致）
    class MouseTrajectoryNet(torch.nn.Module):
        def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, output_points=10):
            super(MouseTrajectoryNet, self).__init__()
            self.input_fc = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2)
            )
            self.lstm = torch.nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )
            self.output_fc = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(hidden_dim // 2, output_points * 2)
            )
        
        def forward(self, x):
            batch_size = x.size(0)
            x = self.input_fc(x)
            x_seq = x.unsqueeze(1).repeat(1, 10, 1)
            lstm_out, _ = self.lstm(x_seq)
            lstm_out = lstm_out[:, -1, :]
            output = self.output_fc(lstm_out)
            output = output.view(batch_size, 10, 2)
            return output
    
    model = MouseTrajectoryNet(input_dim=2, hidden_dim=128, num_layers=2, output_points=10).to(device)
    
    # 尝试加载模型权重
    model_path = 'best_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✓ 模型已加载: {model_path}")
        return True
    else:
        print(f"⚠ 警告: 模型文件 {model_path} 不存在！")
        print("   请先运行 train.py 训练模型。")
        return False

@app.route('/')
def index():
    """API首页"""
    return jsonify({
        'message': '鼠标轨迹预测API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - 预测鼠标轨迹',
            '/health': 'GET - 健康检查'
        },
        'usage': {
            'url': '/predict',
            'method': 'POST',
            'body': {
                'target_point': [100, 200]  # 目标点坐标 [x, y]
            },
            'response': {
                'trajectory': [[x1, y1], [x2, y2], ...],  # 10个关键点
                'target_point': [100, 200]
            }
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """预测鼠标轨迹"""
    if model is None:
        return jsonify({
            'error': '模型未加载',
            'message': '请先运行 train.py 训练模型'
        }), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data or 'target_point' not in data:
            return jsonify({
                'error': '缺少参数',
                'message': '请提供 target_point 参数，格式: {"target_point": [x, y]}'
            }), 400
        
        target_point = data['target_point']
        
        # 验证输入格式
        if not isinstance(target_point, list) or len(target_point) != 2:
            return jsonify({
                'error': '参数格式错误',
                'message': 'target_point 应该是包含两个数字的列表，例如: [100, 200]'
            }), 400
        
        try:
            target_x, target_y = float(target_point[0]), float(target_point[1])
        except (ValueError, TypeError):
            return jsonify({
                'error': '参数类型错误',
                'message': 'target_point 中的值必须是数字'
            }), 400
        
        # 准备输入
        input_tensor = torch.FloatTensor([[target_x, target_y]]).to(device)
        
        # 预测
        with torch.no_grad():
            output = model(input_tensor)
            trajectory = output[0].cpu().numpy().tolist()
        
        # 返回结果
        return jsonify({
            'target_point': [target_x, target_y],
            'trajectory': trajectory,  # 10个关键点的坐标
            'message': '预测成功'
        })
        
    except Exception as e:
        return jsonify({
            'error': '预测失败',
            'message': str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """批量预测鼠标轨迹"""
    if model is None:
        return jsonify({
            'error': '模型未加载',
            'message': '请先运行 train.py 训练模型'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'target_points' not in data:
            return jsonify({
                'error': '缺少参数',
                'message': '请提供 target_points 参数，格式: {"target_points": [[x1, y1], [x2, y2], ...]}'
            }), 400
        
        target_points = data['target_points']
        
        if not isinstance(target_points, list) or len(target_points) == 0:
            return jsonify({
                'error': '参数格式错误',
                'message': 'target_points 应该是包含坐标列表的数组'
            }), 400
        
        # 验证并转换输入
        input_list = []
        for i, point in enumerate(target_points):
            if not isinstance(point, list) or len(point) != 2:
                return jsonify({
                    'error': f'第 {i+1} 个点的格式错误',
                    'message': '每个点应该是 [x, y] 格式'
                }), 400
            try:
                input_list.append([float(point[0]), float(point[1])])
            except (ValueError, TypeError):
                return jsonify({
                    'error': f'第 {i+1} 个点的值类型错误',
                    'message': '坐标值必须是数字'
                }), 400
        
        # 准备输入
        input_tensor = torch.FloatTensor(input_list).to(device)
        
        # 批量预测
        with torch.no_grad():
            output = model(input_tensor)
            trajectories = output.cpu().numpy().tolist()
        
        # 返回结果
        return jsonify({
            'target_points': input_list,
            'trajectories': trajectories,  # 每个目标点对应的10个关键点
            'count': len(trajectories),
            'message': '批量预测成功'
        })
        
    except Exception as e:
        return jsonify({
            'error': '批量预测失败',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 50)
    print("鼠标轨迹预测API")
    print("=" * 50)
    
    # 加载模型
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n⚠ 警告: 模型未加载，API将无法进行预测")
        print("   请先运行: python train.py")
    
    print("\n启动API服务器...")
    print("API地址: http://localhost:5000")
    print("文档: http://localhost:5000")
    print("预测接口: http://localhost:5000/predict")
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)

