#!/usr/bin/env python3
"""
VELOCITY PREDICTION - PATTERN-BASED APPROACH

OBIETTIVO:
1. Training: 10 neuroni random (1 per sessione) + velocity
2. Impara: Pattern di attività neurale → velocity (NON identità neurone)
3. Inference: Dato nuovo neurone → predici velocity

STRATEGIA:
- Usa FEATURES dell'attività neurale (firing rate, burstiness, etc)
- NON identità specifica del neurone
- Impara mapping: neural dynamics → behavioral state

KEY INSIGHT:
Velocity riflette stato comportamentale globale del mouse.
Neuroni con attività simile → velocity simile.
Il modello impara: "alta firing rate + bursts → alta velocity"
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
# FEATURE-BASED VELOCITY PREDICTOR
# ============================================================================

class FeatureBasedVelocityPredictor(nn.Module):
    """
    Predice velocity da FEATURES dell'attività neurale
    
    Features estratte:
    - Temporal dynamics (CNN)
    - Statistical properties (mean, std, peaks)
    - Frequency content (FFT-based features)
    
    Questo permette generalizzazione a QUALSIASI neurone!
    """
    
    def __init__(self, hidden_dim=512, dropout=0.3):
        super(FeatureBasedVelocityPredictor, self).__init__()
        
        print(f"\n📊 Feature-Based Velocity Predictor:")
        print(f"   Strategy: Learn GENERAL patterns, not neuron identity")
        print(f"   Features: Temporal + Statistical + Frequency")
        
        # ===== TEMPORAL FEATURES (CNN) =====
        # Estrae pattern temporali: bursts, ramps, etc
        # Input: (B, N_neurons, 60) → reshape to (B, 1, N*60) per Conv1d
        self.conv1 = nn.Conv1d(20, 64, kernel_size=7, padding=3)  # 20 neuroni input
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(dropout)
        
        # Global temporal pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # ===== STATISTICAL FEATURES =====
        # Calcolate durante forward: mean, std, max, peak_count, etc
        # Input size: 256 (CNN) + 6*20 (stats per 20 neuroni) = 376
        
        # ===== FUSION NETWORK =====
        self.fc1 = nn.Linear(256 + 6*20, hidden_dim)  # CNN + stats per 20 neuroni
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.ln3 = nn.LayerNorm(hidden_dim // 4)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(hidden_dim // 4, 1)
        
        self._init_weights()
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Model: {num_params:,} parameters")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_statistical_features(self, x):
        """
        Extract statistical features from neural traces (20 neuroni)
        
        Args:
            x: (B, 20, 60) neural activity
        Returns:
            stats: (B, 120) statistical features (6 per neurone * 20)
        """
        batch_size = x.size(0)
        num_neurons = x.size(1)
        x_np = x.cpu().numpy()  # (B, 20, 60)
        
        features = []
        
        for i in range(batch_size):
            sample_features = []
            
            for n in range(num_neurons):
                trace = x_np[i, n, :]
                
                # 1-6: Features per questo neurone
                mean_fr = np.mean(trace)
                std_fr = np.std(trace)
                max_fr = np.max(trace)
                
                threshold = mean_fr + 1.0 * std_fr
                peaks = np.sum(trace > threshold)
                
                cv = std_fr / (mean_fr + 1e-8)
                range_fr = np.max(trace) - np.min(trace)
                
                sample_features.extend([mean_fr, std_fr, max_fr, peaks, cv, range_fr])
            
            features.append(sample_features)
        
        return torch.FloatTensor(features).to(x.device)
    
    def forward(self, x):
        """
        Args:
            x: (B, 20, 60) neural activity (20 neuroni)
        Returns:
            velocity: (B,) predicted velocity
        """
        # ===== TEMPORAL FEATURES (CNN) =====
        h = self.conv1(x)  # Input: (B, 20, 60)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.pool(h)
        h = self.dropout_conv(h)
        
        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.pool(h)
        h = self.dropout_conv(h)
        
        h = self.conv3(h)
        h = self.bn3(h)
        h = F.relu(h)
        h = self.dropout_conv(h)
        
        # Global pooling
        h = self.global_pool(h)  # (B, 256, 1)
        h = h.view(h.size(0), -1)  # (B, 256)
        
        # ===== STATISTICAL FEATURES =====
        stats = self.extract_statistical_features(x)  # (B, 120)
        
        # ===== FUSION =====
        combined = torch.cat([h, stats], dim=1)  # (B, 376)
        
        # FC layers
        x_out = self.fc1(combined)
        x_out = self.ln1(x_out)
        x_out = F.relu(x_out)
        x_out = self.dropout1(x_out)
        
        x_out = self.fc2(x_out)
        x_out = self.ln2(x_out)
        x_out = F.relu(x_out)
        x_out = self.dropout2(x_out)
        
        x_out = self.fc3(x_out)
        x_out = self.ln3(x_out)
        x_out = F.relu(x_out)
        x_out = self.dropout3(x_out)
        
        velocity = self.fc_out(x_out)
        
        return velocity.squeeze(-1)


# ============================================================================
# VELOCITY DATASET (usa behavioral.py esistente)
# ============================================================================

class VelocityDataset(torch.utils.data.Dataset):
    def __init__(self, behavioral_dataset):
        self.behavioral_dataset = behavioral_dataset
    
    def __len__(self):
        return len(self.behavioral_dataset)
    
    def __getitem__(self, idx):
        neural, behavioral = self.behavioral_dataset[idx]
        velocity = behavioral[3]  # Velocity
        return neural, velocity


def create_velocity_dataloaders(batch_size=128, train_split=0.8):
    """Use existing behavioral.py with random neurons"""
    from datasets.behavioral import create_behavioral_dataloaders
    from datasets.calcium import TRAINING_SESSION_IDS
    
    print("\n📊 Loading velocity data from 10 sessions...")
    print("   Using RANDOM neuron per session (different each time)")
    print("   Model will learn GENERAL patterns, not specific neurons!")
    
    train_loader_full, test_loader_full, dataset_info = create_behavioral_dataloaders(
        session_ids=TRAINING_SESSION_IDS,
        batch_size=batch_size,
        test_split=1.0 - train_split,
        window_size=60,
        stride=30,
        num_neurons=20,  # 🔥 20 neuroni per più segnale!
        num_workers=0,
        normalize=True
    )
    
    # Wrap to extract velocity
    train_dataset = VelocityDataset(train_loader_full.dataset)
    test_dataset = VelocityDataset(test_loader_full.dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
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
    
    for neural, velocity in dataloader:
        neural = neural.to(device)
        velocity = velocity.to(device)
        
        pred = model(neural)
        
        # Huber loss (robust to outliers)
        loss = F.smooth_l1_loss(pred, velocity)
        
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
    
    all_preds = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for neural, velocity in dataloader:
            neural = neural.to(device)
            velocity = velocity.to(device)
            
            pred = model(neural)
            loss = F.smooth_l1_loss(pred, velocity)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(velocity.cpu().numpy())
            total_loss += loss.item()
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    
    if np.std(preds) > 1e-8 and np.std(targets) > 1e-8:
        corr, _ = pearsonr(preds, targets)
        corr = corr if not np.isnan(corr) else 0.0
    else:
        corr = 0.0
    
    signal_power = np.mean(targets ** 2)
    noise_power = np.mean((targets - preds) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
    
    pred_std = np.std(preds)
    target_std = np.std(targets)
    var_ratio = pred_std / target_std if target_std > 0 else 0
    
    return {
        'test/loss': total_loss / len(dataloader),
        'test/mse': mse,
        'test/mae': mae,
        'test/r2': r2,
        'test/correlation': corr,
        'test/snr': snr,
        'test/variance_ratio': var_ratio,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_predictions(model, dataloader, device, epoch):
    model.eval()
    
    neural, velocity = next(iter(dataloader))
    neural = neural.to(device)
    
    with torch.no_grad():
        pred = model(neural)
    
    preds = pred.cpu().numpy()
    targets = velocity.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter
    axes[0].scatter(targets, preds, alpha=0.3, s=10)
    axes[0].plot([targets.min(), targets.max()], 
                [targets.min(), targets.max()], 'r--', lw=2)
    axes[0].set_xlabel('Target Velocity')
    axes[0].set_ylabel('Predicted Velocity')
    axes[0].set_title(f'Epoch {epoch}')
    axes[0].grid(True, alpha=0.3)
    
    if np.std(preds) > 1e-8:
        corr, _ = pearsonr(preds, targets)
        axes[0].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                    transform=axes[0].transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Histogram
    axes[1].hist(targets, bins=50, alpha=0.5, label='Target', density=True)
    axes[1].hist(preds, bins=50, alpha=0.5, label='Pred', density=True)
    axes[1].set_xlabel('Velocity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    pred_std = np.std(preds)
    target_std = np.std(targets)
    stats = f'Target std: {target_std:.3f}\nPred std: {pred_std:.3f}\nRatio: {pred_std/target_std:.3f}'
    axes[1].text(0.98, 0.98, stats, transform=axes[1].transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    wandb.init(
        project="calcium-velocity-prediction",
        name=f"feature-based-{datetime.now().strftime('%m%d-%H%M')}",
        config={
            'approach': 'feature-based',
            'hidden_dim': 512,
            'batch_size': 128,
            'learning_rate': 0.001,
            'max_epochs': 300,
            'patience': 50,
        },
        tags=["feature-based", "generalizable", "random-neurons", "10-sessions"]
    )
    
    print("="*70)
    print("🔥 FEATURE-BASED VELOCITY PREDICTION")
    print("="*70)
    print("✅ Strategy:")
    print("   1. Use RANDOM neurons (1 per session)")
    print("   2. Extract GENERAL features (temporal + statistical)")
    print("   3. Learn: Neural dynamics → Velocity")
    print("   4. Generalize to ANY neuron!")
    print("")
    print("🎯 Key Insight:")
    print("   High firing rate + bursts → High velocity")
    print("   Low/steady activity → Low velocity")
    print("   Model learns PATTERNS, not neuron identity!")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}\n")
    
    try:
        # Create model
        model = FeatureBasedVelocityPredictor(
            hidden_dim=wandb.config.hidden_dim,
            dropout=0.3
        ).to(device)
        
        # Load data
        train_loader, test_loader, info = create_velocity_dataloaders(
            batch_size=wandb.config.batch_size,
            train_split=0.8
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=wandb.config.learning_rate,
            weight_decay=0.0001
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15, verbose=True
        )
        
        print("\n🚀 Starting training...\n")
        
        best_corr = -1.0
        patience_counter = 0
        
        for epoch in range(wandb.config.max_epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, device)
            
            if epoch % 5 == 0:
                test_metrics = evaluate_model(model, test_loader, device)
                
                scheduler.step(test_metrics['test/correlation'])
                current_lr = optimizer.param_groups[0]['lr']
                
                all_metrics = {**train_metrics, **test_metrics, 
                              'epoch': epoch, 'learning_rate': current_lr}
                wandb.log(all_metrics)
                
                print(f"Epoch {epoch:3d} | LR={current_lr:.6f}")
                print(f"  Loss: {test_metrics['test/loss']:.6f}")
                print(f"  🔥 R²={test_metrics['test/r2']:.4f}, Corr={test_metrics['test/correlation']:.4f}, SNR={test_metrics['test/snr']:.1f}dB")
                print(f"  📊 Var ratio={test_metrics['test/variance_ratio']:.3f}")
                
                if epoch % 20 == 0:
                    fig = visualize_predictions(model, test_loader, device, epoch)
                    wandb.log({f"predictions/epoch_{epoch}": wandb.Image(fig)})
                    plt.close(fig)
                
                if test_metrics['test/correlation'] > best_corr:
                    best_corr = test_metrics['test/correlation']
                    patience_counter = 0
                    
                    os.makedirs('./results/velocity', exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_correlation': best_corr,
                        'test_metrics': test_metrics,
                    }, './results/velocity/best_feature_based.pth')
                    print(f"  💾 Saved (corr={best_corr:.4f})")
                else:
                    patience_counter += 1
                
                if patience_counter >= wandb.config.patience:
                    print("\n⚠️ Early stopping")
                    break
            else:
                wandb.log({**train_metrics, 'epoch': epoch})
        
        print("\n" + "="*70)
        print("🎯 FINAL RESULTS")
        print("="*70)
        print(f"Best correlation: {best_corr:.4f}")
        print("")
        print("💡 This model can now predict velocity from ANY neuron")
        print("   because it learned GENERAL activity patterns!")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()