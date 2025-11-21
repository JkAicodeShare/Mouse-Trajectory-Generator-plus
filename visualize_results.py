"""
Training Results Visualization Tool
Display training loss curves, prediction comparisons, etc.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import pandas as pd
import os

# Define model class (to avoid import issues)
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

# Define data parsing functions
def parse_enhanced_data(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None, None
    df = pd.read_csv(csv_path, header=None)
    if len(df) == 0:
        return None, None
    targets = []
    key_points = []
    for idx, row in df.iterrows():
        target_str = str(row.iloc[0])
        parts = target_str.split(',')
        if len(parts) >= 2:
            targets.append([float(parts[0]), float(parts[1])])
        else:
            continue
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

def parse_legacy_data(csv_path):
    return parse_enhanced_data(csv_path)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

def load_model_and_data():
    """Load model and data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MouseTrajectoryNet(input_dim=2, hidden_dim=128, num_layers=2, output_points=10).to(device)
    model_path = 'best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file {model_path} not found!")
        print("   Please run python train.py to train the model first")
        return None, None, None, None
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Model loaded: {model_path}")
    
    # Load data
    train_csv_path = 'mouse_data.csv'
    test_csv_path = 'mouse_data_test.csv'
    
    train_targets, train_key_points = parse_enhanced_data(train_csv_path)
    if train_targets is None:
        train_targets, train_key_points = parse_legacy_data(train_csv_path)
    
    test_targets, test_key_points = parse_legacy_data(test_csv_path)
    if test_targets is None:
        test_targets, test_key_points = parse_enhanced_data(test_csv_path)
    
    if train_targets is None or len(train_targets) == 0:
        print("❌ Error: Training data is empty!")
        return None, None, None, None
    
    if test_targets is None or len(test_targets) == 0:
        test_targets = train_targets
        test_key_points = train_key_points
    
    print(f"✓ Training samples: {len(train_targets)}")
    print(f"✓ Test samples: {len(test_targets)}")
    
    return model, device, test_targets, test_key_points

def visualize_trajectory_comparison(model, device, test_targets, test_key_points, num_samples=6):
    """Visualize trajectory comparison: True trajectory vs Predicted trajectory"""
    model.eval()
    
    # Select samples for visualization
    num_samples = min(num_samples, len(test_targets))
    indices = np.linspace(0, len(test_targets) - 1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Prediction Comparison\nTrue Trajectory vs Predicted Trajectory', fontsize=20, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    with torch.no_grad():
        for idx, ax_idx in enumerate(indices):
            ax = axes[idx]
            
            # Get true data
            target = test_targets[ax_idx]
            true_trajectory = test_key_points[ax_idx]
            
            # Predict
            input_tensor = torch.FloatTensor([target]).to(device)
            pred_trajectory = model(input_tensor)[0].cpu().numpy()
            
            # Plot starting point (target point)
            ax.scatter([target[0]], [target[1]], c='green', s=200, marker='*', 
                      label='Target Point', zorder=5, edgecolors='black', linewidths=2)
            
            # Plot true trajectory
            true_x = [0] + true_trajectory[:, 0].tolist()
            true_y = [0] + true_trajectory[:, 1].tolist()
            ax.plot(true_x, true_y, 'o-', color=colors[0], linewidth=3, 
                   markersize=8, label='True Trajectory', alpha=0.8, zorder=3)
            
            # Plot predicted trajectory
            pred_x = [0] + pred_trajectory[:, 0].tolist()
            pred_y = [0] + pred_trajectory[:, 1].tolist()
            ax.plot(pred_x, pred_y, 's-', color=colors[1], linewidth=3, 
                   markersize=8, label='Predicted Trajectory', alpha=0.8, zorder=4)
            
            # Calculate error
            mse = np.mean((true_trajectory - pred_trajectory) ** 2)
            
            # Set title and labels
            ax.set_title(f'Sample {ax_idx+1}\nMSE: {mse:.2f}', fontsize=12, fontweight='bold')
            ax.set_xlabel('X Coordinate', fontsize=11)
            ax.set_ylabel('Y Coordinate', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            ax.set_aspect('equal', adjustable='box')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Prediction comparison saved: prediction_comparison.png")
    plt.show()

def visualize_trajectory_3d(model, device, test_targets, test_key_points, num_samples=3):
    """3D trajectory visualization"""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        model.eval()
        num_samples = min(num_samples, len(test_targets))
        indices = np.linspace(0, len(test_targets) - 1, num_samples, dtype=int)
        
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle('3D Trajectory Visualization', fontsize=16, fontweight='bold')
        
        with torch.no_grad():
            for idx, sample_idx in enumerate(indices):
                ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
                
                target = test_targets[sample_idx]
                true_trajectory = test_key_points[sample_idx]
                
                input_tensor = torch.FloatTensor([target]).to(device)
                pred_trajectory = model(input_tensor)[0].cpu().numpy()
                
                # Create time axis
                time_steps = np.arange(len(true_trajectory) + 1)
                
                # True trajectory
                true_x = [0] + true_trajectory[:, 0].tolist()
                true_y = [0] + true_trajectory[:, 1].tolist()
                ax.plot(true_x, true_y, time_steps, 'o-', color=colors[0], 
                       linewidth=2, markersize=6, label='True Trajectory', alpha=0.8)
                
                # Predicted trajectory
                pred_x = [0] + pred_trajectory[:, 0].tolist()
                pred_y = [0] + pred_trajectory[:, 1].tolist()
                ax.plot(pred_x, pred_y, time_steps, 's-', color=colors[1], 
                       linewidth=2, markersize=6, label='Predicted Trajectory', alpha=0.8)
                
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.set_zlabel('Time Step')
                ax.set_title(f'Sample {sample_idx+1}')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('trajectory_3d.png', dpi=300, bbox_inches='tight')
        print("✓ 3D trajectory saved: trajectory_3d.png")
        plt.show()
    except ImportError:
        print("⚠ Skipping 3D visualization (requires matplotlib 3D support)")

def visualize_error_distribution(model, device, test_targets, test_key_points):
    """Visualize error distribution"""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for i in range(len(test_targets)):
            target = test_targets[i]
            true_trajectory = test_key_points[i]
            
            input_tensor = torch.FloatTensor([target]).to(device)
            pred_trajectory = model(input_tensor)[0].cpu().numpy()
            
            # Calculate error for each point
            point_errors = np.sqrt(np.sum((true_trajectory - pred_trajectory) ** 2, axis=1))
            errors.extend(point_errors.tolist())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Prediction Error Analysis', fontsize=16, fontweight='bold')
    
    # Error histogram
    ax1.hist(errors, bins=30, color=colors[2], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Error (pixels)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Error Distribution Histogram', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(errors):.2f}')
    ax1.axvline(np.median(errors), color='blue', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(errors):.2f}')
    ax1.legend()
    
    # Error boxplot
    box_data = [errors]
    bp = ax2.boxplot(box_data, patch_artist=True, labels=['Prediction Error'])
    bp['boxes'][0].set_facecolor(colors[3])
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel('Error (pixels)', fontsize=12)
    ax2.set_title('Error Boxplot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Mean: {np.mean(errors):.2f}\n'
    stats_text += f'Median: {np.median(errors):.2f}\n'
    stats_text += f'Std: {np.std(errors):.2f}\n'
    stats_text += f'Max: {np.max(errors):.2f}\n'
    stats_text += f'Min: {np.min(errors):.2f}'
    ax2.text(0.7, 0.95, stats_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Error distribution saved: error_distribution.png")
    plt.show()

def visualize_statistics(model, device, test_targets, test_key_points):
    """Visualize statistics"""
    model.eval()
    mse_list = []
    mae_list = []
    
    with torch.no_grad():
        for i in range(len(test_targets)):
            target = test_targets[i]
            true_trajectory = test_key_points[i]
            
            input_tensor = torch.FloatTensor([target]).to(device)
            pred_trajectory = model(input_tensor)[0].cpu().numpy()
            
            mse = np.mean((true_trajectory - pred_trajectory) ** 2)
            mae = np.mean(np.abs(true_trajectory - pred_trajectory))
            
            mse_list.append(mse)
            mae_list.append(mae)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Performance Statistics', fontsize=16, fontweight='bold')
    
    # MSE distribution
    axes[0].bar(range(len(mse_list)), mse_list, color=colors[0], alpha=0.7, edgecolor='black')
    axes[0].axhline(np.mean(mse_list), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean MSE: {np.mean(mse_list):.2f}')
    axes[0].set_xlabel('Sample Index', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('Mean Squared Error (MSE) Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE distribution
    axes[1].bar(range(len(mae_list)), mae_list, color=colors[1], alpha=0.7, edgecolor='black')
    axes[1].axhline(np.mean(mae_list), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean MAE: {np.mean(mae_list):.2f}')
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Mean Absolute Error (MAE) Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_statistics.png', dpi=300, bbox_inches='tight')
    print("✓ Statistics saved: model_statistics.png")
    plt.show()
    
    # Print statistics
    print("\n" + "="*50)
    print("Model Performance Statistics")
    print("="*50)
    print(f"Test samples: {len(test_targets)}")
    print(f"Mean MSE: {np.mean(mse_list):.2f}")
    print(f"Mean MAE: {np.mean(mae_list):.2f}")
    print(f"MSE Std: {np.std(mse_list):.2f}")
    print(f"MAE Std: {np.std(mae_list):.2f}")
    print("="*50)

def main():
    """Main function"""
    print("="*60)
    print("Mouse Trajectory Prediction - Training Results Visualization")
    print("="*60)
    
    # Load model and data
    model, device, test_targets, test_key_points = load_model_and_data()
    
    if model is None:
        return
    
    print("\nGenerating visualization charts...")
    
    # 1. Trajectory comparison
    print("\n1. Generating prediction comparison chart...")
    visualize_trajectory_comparison(model, device, test_targets, test_key_points, num_samples=6)
    
    # 2. Error distribution
    print("\n2. Generating error distribution chart...")
    visualize_error_distribution(model, device, test_targets, test_key_points)
    
    # 3. Statistics
    print("\n3. Generating statistics chart...")
    visualize_statistics(model, device, test_targets, test_key_points)
    
    # 4. 3D visualization (optional)
    print("\n4. Generating 3D trajectory chart...")
    visualize_trajectory_3d(model, device, test_targets, test_key_points, num_samples=3)
    
    print("\n" + "="*60)
    print("✓ All visualization charts generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - prediction_comparison.png  (Prediction comparison)")
    print("  - error_distribution.png    (Error distribution)")
    print("  - model_statistics.png      (Statistics)")
    print("  - trajectory_3d.png         (3D trajectory)")

if __name__ == '__main__':
    main()
