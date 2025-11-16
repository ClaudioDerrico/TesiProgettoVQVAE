#!/usr/bin/env python3
"""
VELOCITY PREDICTION - END-TO-END FROM SCRATCH

SOLUZIONE RADICALE al mode collapse:
- NO encoder pretrainato
- NO VQ-VAE
- Simple CNN + LSTM → Velocity
- Train tutto end-to-end

Approccio: Più semplice, più diretto, più probabilità di funzionare
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr

# ============================================================================
# SIMPLE END-TO-END MODEL - NO VQ-VAE
# ============================================================================

class SimpleVelocityPredictor(nn.Module):
    """
    Simple CNN + LSTM per predire velocity da neural activity
    
    Architecture:
    Neural (1, 60) → CNN → LSTM → FC → Velocity (1,)
    
    NO quantization, NO pretrained encoder
    Just learn the mapping directly!
    """
    
    def __init__(self, hidden_dim=256, num_layers=2, dropout=0.3):
        super(SimpleVelocityPredictor, self).__init__()
        
        print("\n📊 Simple End-to-End Velocity Predictor:")
        print(f"   Input: (1, 60) neural activity")
        print(f"   CNN: 1→32→64 channels")
        print(f"   LSTM: {num_layers} layers, hidden={hidden_dim}")
        print(f"   FC: {hidden_dim}→128→1")
        print(f"   Output: velocity")
        
        # CNN layers (extract temporal features)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(dropout)
        
        # LSTM (temporal modeling)
        # After 3 pools: 60 → 30 → 15 → 7
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # FC layers
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(64, 1)
        
        # Weight initialization
        self._init_weights()
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Model created: {num_params:,} parameters (ALL trainable!)")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, 60) neural activity
        Returns:
            velocity: (B,) predicted velocity
        """
        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 60 → 30
        x = self.dropout_conv(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 30 → 15
        x = self.dropout_conv(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 15 → 7
        x = self.dropout_conv(x)
        
        # LSTM temporal modeling
        # x shape: (B, 128, 7) → transpose → (B, 7, 128)
        x = x.transpose(1, 2)
        
        # LSTM output: (B, 7, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        x = h_n[-1]  # (B, hidden_dim)
        
        # FC layers
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        velocity = self.fc_out(x)
        
        return velocity.squeeze(-1)


# ============================================================================
# VELOCITY DATASET
# ============================================================================

class VelocityDataset(torch.utils.data.Dataset):
    def __init__(self, behavioral_dataset):
        self.behavioral_dataset = behavioral_dataset
    
    def __len__(self):
        return len(self.behavioral_dataset)
    
    def __getitem__(self, idx):
        neural, behavioral = self.behavioral_dataset[idx]
        velocity = behavioral[3]
        return neural, velocity


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.3,
}

TRAIN_CONFIG = {
    'batch_size': 128,
    'learning_rate': 0.001,  # Start conservative
    'max_epochs': 300,
    'patience': 50,
    'weight_decay': 0.0001,
    'visualize_every': 20,
    'evaluate_every': 5,
}

# ============================================================================
# DATA LOADING
# ============================================================================

def create_velocity_dataloaders(batch_size=128, train_split=0.8):
    from datasets.behavioral import create_behavioral_dataloaders
    from datasets.calcium import TRAINING_SESSION_IDS
    
    print("\n📊 Loading velocity data from ALL 10 sessions...")
    
    train_loader_full, test_loader_full, dataset_info = create_behavioral_dataloaders(
        session_ids=TRAINING_SESSION_IDS,
        batch_size=batch_size,
        test_split=1.0 - train_split,
        window_size=60,
        stride=30,
        num_neurons=1,
        num_workers=0,
        normalize=True
    )
    
    train_dataset_velocity = VelocityDataset(train_loader_full.dataset)
    test_dataset_velocity = VelocityDataset(test_loader_full.dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset_velocity, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset_velocity, batch_size=batch_size, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    
    print(f"✅ Data loaded:")
    print(f"   Train: {dataset_info['train_samples']}")
    print(f"   Test: {dataset_info['test_samples']}")
    
    return train_loader, test_loader, dataset_info

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for neural_data, velocity_target in dataloader:
        neural_data = neural_data.to(device)
        velocity_target = velocity_target.to(device)
        
        velocity_pred = model(neural_data)
        
        # Huber loss (robust to outliers)
        loss = F.smooth_l1_loss(velocity_pred, velocity_target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return {'train/loss': total_loss / len(dataloader)}

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, device):
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for neural_data, velocity_target in dataloader:
            neural_data = neural_data.to(device)
            velocity_target = velocity_target.to(device)
            
            velocity_pred = model(neural_data)
            loss = F.smooth_l1_loss(velocity_pred, velocity_target)
            
            all_predictions.append(velocity_pred.cpu().numpy())
            all_targets.append(velocity_target.cpu().numpy())
            total_loss += loss.item()
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # Metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    
    if np.std(predictions) > 1e-8 and np.std(targets) > 1e-8:
        corr, _ = pearsonr(predictions, targets)
        corr = corr if not np.isnan(corr) else 0.0
    else:
        corr = 0.0
    
    signal_power = np.mean(targets ** 2)
    noise_power = np.mean((targets - predictions) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
    
    pred_std = np.std(predictions)
    target_std = np.std(targets)
    variance_ratio = pred_std / target_std if target_std > 0 else 0
    
    return {
        'test/loss': total_loss / len(dataloader),
        'test/mse': mse,
        'test/mae': mae,
        'test/r2': r2,
        'test/correlation': corr,
        'test/snr': snr,
        'test/pred_std': pred_std,
        'test/target_std': target_std,
        'test/variance_ratio': variance_ratio,
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_predictions(model, dataloader, device, epoch):
    model.eval()
    
    neural_data, velocity_target = next(iter(dataloader))
    neural_data = neural_data.to(device)
    velocity_target = velocity_target.to(device)
    
    with torch.no_grad():
        velocity_pred = model(neural_data)
    
    predictions = velocity_pred.cpu().numpy()
    targets = velocity_target.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter
    axes[0].scatter(targets, predictions, alpha=0.3, s=10)
    axes[0].plot([targets.min(), targets.max()], 
                [targets.min(), targets.max()], 'r--', lw=2)
    axes[0].set_xlabel('Target Velocity')
    axes[0].set_ylabel('Predicted Velocity')
    axes[0].set_title(f'Predictions vs Targets - Epoch {epoch}')
    axes[0].grid(True, alpha=0.3)
    
    # Add correlation text
    if np.std(predictions) > 1e-8 and np.std(targets) > 1e-8:
        corr, _ = pearsonr(predictions, targets)
        axes[0].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                    transform=axes[0].transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Histogram
    axes[1].hist(targets, bins=50, alpha=0.5, label='Targets', density=True, color='blue')
    axes[1].hist(predictions, bins=50, alpha=0.5, label='Predictions', density=True, color='red')
    axes[1].set_xlabel('Velocity')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Distribution - Epoch {epoch}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Stats
    pred_std = np.std(predictions)
    target_std = np.std(targets)
    stats_text = f'Target std: {target_std:.3f}\nPred std: {pred_std:.3f}\nRatio: {pred_std/target_std:.3f}'
    axes[1].text(0.98, 0.98, stats_text, transform=axes[1].transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    wandb.init(
        project="calcium-velocity-prediction",
        name=f"simple-end2end-{datetime.now().strftime('%m%d-%H%M')}",
        config={**MODEL_CONFIG, **TRAIN_CONFIG},
        tags=["end-to-end", "no-vqvae", "cnn-lstm", "10-sessions"]
    )
    
    print("="*70)
    print("🔥 SIMPLE END-TO-END VELOCITY PREDICTION")
    print("="*70)
    print("✅ Key features:")
    print("   - NO pretrained encoder")
    print("   - NO VQ-VAE quantization")
    print("   - Simple CNN + LSTM architecture")
    print("   - Train everything end-to-end")
    print("   - ALL parameters trainable")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}\n")
    
    try:
        # Create model
        print("🏗️  Creating simple end-to-end model...")
        model = SimpleVelocityPredictor(
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        ).to(device)
        
        # Load data
        train_loader, test_loader, dataset_info = create_velocity_dataloaders(
            batch_size=TRAIN_CONFIG['batch_size'],
            train_split=0.8
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15, verbose=True
        )
        
        print("\n🚀 Starting training...\n")
        
        best_corr = -1.0
        patience_counter = 0
        
        for epoch in range(TRAIN_CONFIG['max_epochs']):
            train_metrics = train_epoch(model, train_loader, optimizer, device)
            
            if epoch % TRAIN_CONFIG['evaluate_every'] == 0:
                test_metrics = evaluate_model(model, test_loader, device)
                
                scheduler.step(test_metrics['test/correlation'])
                current_lr = optimizer.param_groups[0]['lr']
                
                all_metrics = {**train_metrics, **test_metrics, 'epoch': epoch, 'learning_rate': current_lr}
                wandb.log(all_metrics)
                
                print(f"Epoch {epoch:3d} | LR={current_lr:.6f}")
                print(f"  Loss: {test_metrics['test/loss']:.6f}")
                print(f"  🔥 R²={test_metrics['test/r2']:.4f}, Corr={test_metrics['test/correlation']:.4f}, SNR={test_metrics['test/snr']:.1f}dB")
                print(f"  📊 Var ratio={test_metrics['test/variance_ratio']:.3f} (target: ~1.0)")
                
                if epoch % TRAIN_CONFIG['visualize_every'] == 0:
                    fig = visualize_predictions(model, test_loader, device, epoch)
                    wandb.log({f"predictions/epoch_{epoch}": wandb.Image(fig)})
                    plt.close(fig)
                
                current_corr = test_metrics['test/correlation']
                if current_corr > best_corr:
                    best_corr = current_corr
                    patience_counter = 0
                    
                    os.makedirs('./results/velocity', exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_correlation': best_corr,
                        'test_metrics': test_metrics,
                        'model_config': MODEL_CONFIG,
                    }, './results/velocity/best_simple_end2end.pth')
                    print(f"  💾 Saved (corr={best_corr:.4f})")
                else:
                    patience_counter += 1
                
                if patience_counter >= TRAIN_CONFIG['patience']:
                    print(f"\n⚠️ Early stopping at epoch {epoch}")
                    break
            else:
                wandb.log({**train_metrics, 'epoch': epoch})
        
        print("\n" + "="*70)
        print("🎯 FINAL EVALUATION")
        print("="*70)
        
        final_metrics = evaluate_model(model, test_loader, device)
        
        print(f"Test Loss: {final_metrics['test/loss']:.6f}")
        print(f"R²: {final_metrics['test/r2']:.4f}")
        print(f"Correlation: {final_metrics['test/correlation']:.4f}")
        print(f"SNR: {final_metrics['test/snr']:.1f} dB")
        print(f"Variance Ratio: {final_metrics['test/variance_ratio']:.3f}")
        
        wandb.log({
            'final/loss': final_metrics['test/loss'],
            'final/r2': final_metrics['test/r2'],
            'final/correlation': final_metrics['test/correlation'],
            'final/snr': final_metrics['test/snr'],
        })
        
        print("\n✅ Training completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()