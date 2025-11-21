"""
简单的API测试客户端
演示如何使用鼠标轨迹预测API
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    """测试健康检查"""
    print("\n1. 测试健康检查...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        print(f"   请确保API服务器正在运行: python test_api.py")
        return False

def test_predict():
    """测试单个预测"""
    print("\n2. 测试单个预测...")
    try:
        data = {
            "target_point": [100, 200]
        }
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print(f"   状态码: {response.status_code}")
        result = response.json()
        print(f"   目标点: {result.get('target_point', 'N/A')}")
        print(f"   预测轨迹点数: {len(result.get('trajectory', []))}")
        print(f"   前3个点: {result.get('trajectory', [])[:3]}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False

def test_predict_batch():
    """测试批量预测"""
    print("\n3. 测试批量预测...")
    try:
        data = {
            "target_points": [
                [50, 100],
                [150, 200],
                [200, 300]
            ]
        }
        response = requests.post(
            f"{API_URL}/predict_batch",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print(f"   状态码: {response.status_code}")
        result = response.json()
        print(f"   批量预测数量: {result.get('count', 0)}")
        print(f"   每个轨迹的点数: {len(result.get('trajectories', [[]])[0])}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False

def test_error_cases():
    """测试错误情况"""
    print("\n4. 测试错误处理...")
    
    # 测试缺少参数
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={},
            headers={"Content-Type": "application/json"}
        )
        print(f"   缺少参数 - 状态码: {response.status_code}")
        print(f"   错误信息: {response.json().get('message', 'N/A')}")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    # 测试格式错误
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"target_point": [100]},  # 只有1个值
            headers={"Content-Type": "application/json"}
        )
        print(f"   格式错误 - 状态码: {response.status_code}")
        print(f"   错误信息: {response.json().get('message', 'N/A')}")
    except Exception as e:
        print(f"   ❌ 错误: {e}")

if __name__ == '__main__':
    print("=" * 50)
    print("鼠标轨迹预测API - 测试客户端")
    print("=" * 50)
    
    # 运行测试
    if test_health():
        test_predict()
        test_predict_batch()
        test_error_cases()
    else:
        print("\n⚠ 无法连接到API服务器")
        print("   请先启动API服务器: python test_api.py")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)

