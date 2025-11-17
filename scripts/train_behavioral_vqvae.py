#!/usr/bin/env python3
"""
Training Script for Behavioral VQ-VAE - VELOCITY ONLY

Uses pretrained neural encoder + codebook (frozen)
Trains only velocity decoder to predict running velocity

Transfer learning approach:
1. Load pretrained VQ-VAE (neural reconstruction)
2. Freeze encoder + codebook
3. Train new decoder for velocity prediction
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

from models.behavioral_vqvae import create_behavioral_model_from_checkpoint
from models.vqvae import CalciumVQVAE
from models.dual_vqvae import DualCalciumVQVAE
from datasets.velocity import create_velocity_dataloaders  # ✅ New import
from datasets.calcium import TRAINING_SESSION_IDS

# ============================================================================
# CONFIGURATION
# ============================================================================

PRETRAINED_CONFIG = {
    'checkpoint_path': 'results/best_single_neuron_model_32.pth',
    'model_class': CalciumVQVAE,
    
    # Model architecture (must match pretrained)
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 32,
    'embedding_dim': 32,
    'commitment_cost': 2.0,
    'dropout_rate': 0.1,
    'use_quantizer': True,
}

BEHAVIORAL_CONFIG = {
    'freeze_encoder': True,
    'freeze_codebook': True,
    'hidden_dim': 256,
    'dropout_rate': 0.3,
}

TRAIN_CONFIG = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'max_epochs': 200,
    'patience': 30,
    'weight_decay': 0.0001,
    'visualize_every': 10,
    'evaluate_every': 5,
}

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Train one epoch"""
    model.train()
    model.velocity_decoder.train()  # ✅ Changed from behavioral_decoder
    model.encoder.eval()
    model.vector_quantization.eval()
    
    total_loss = 0
    
    for neural_data, velocity_targets in dataloader:  # ✅ Only velocity now
        neural_data = neural_data.to(device)
        velocity_targets = velocity_targets.to(device)
        
        # Forward pass
        predictions = model(neural_data, return_quantized=False)
        
        # Loss (Huber loss - robust to outliers)
        loss = F.smooth_l1_loss(predictions, velocity_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.velocity_decoder.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return {'train/loss': total_loss / len(dataloader)}


def evaluate_model(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for neural_data, velocity_targets in dataloader:
            neural_data = neural_data.to(device)
            velocity_targets = velocity_targets.to(device)
            
            predictions = model(neural_data, return_quantized=False)
            loss = F.smooth_l1_loss(predictions, velocity_targets)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(velocity_targets.cpu().numpy())
            total_loss += loss.item()
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # Compute metrics
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
    
    # Signal-to-noise ratio
    signal_power = np.mean(targets ** 2)
    noise_power = np.mean((targets - predictions) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
    
    # Variance ratio
    pred_std = np.std(predictions)
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


def visualize_predictions(model, dataloader, device, epoch):
    """Visualize predictions vs targets"""
    model.eval()
    
    neural_data, velocity_targets = next(iter(dataloader))
    neural_data = neural_data.to(device)
    
    with torch.no_grad():
        predictions = model(neural_data)
    
    predictions = predictions.cpu().numpy()
    targets = velocity_targets.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(targets, predictions, alpha=0.3, s=10)
    axes[0].plot([targets.min(), targets.max()], 
                [targets.min(), targets.max()], 'r--', lw=2, label='Perfect')
    axes[0].set_xlabel('Target Velocity')
    axes[0].set_ylabel('Predicted Velocity')
    axes[0].set_title(f'Epoch {epoch}: Predictions vs Targets')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if np.std(predictions) > 1e-8:
        corr, _ = pearsonr(predictions, targets)
        axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[0].transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Distribution comparison
    axes[1].hist(targets, bins=50, alpha=0.5, label='Target', density=True)
    axes[1].hist(predictions, bins=50, alpha=0.5, label='Predicted', density=True)
    axes[1].set_xlabel('Velocity')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    pred_std = np.std(predictions)
    target_std = np.std(targets)
    stats_text = f'Target std: {target_std:.3f}\nPred std: {pred_std:.3f}\nRatio: {pred_std/target_std:.3f}'
    axes[1].text(0.98, 0.98, stats_text, transform=axes[1].transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    wandb.init(
        project="calcium-velocity-vqvae",
        name=f"velocity-transfer-{datetime.now().strftime('%m%d-%H%M')}",
        config={
            **PRETRAINED_CONFIG,
            **BEHAVIORAL_CONFIG,
            **TRAIN_CONFIG
        },
        tags=["velocity", "transfer-learning", "frozen-encoder", "single-neuron"]
    )
    
    print("="*70)
    print("🚀 VELOCITY PREDICTION - TRANSFER LEARNING APPROACH")
    print("="*70)
    print("📊 Strategy:")
    print("   1. Load pretrained VQ-VAE (neural reconstruction)")
    print("   2. Freeze encoder + codebook")
    print("   3. Train NEW decoder for velocity prediction")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}\n")
    
    try:
        # Check if checkpoint exists
        if not os.path.exists(PRETRAINED_CONFIG['checkpoint_path']):
            print(f"❌ Checkpoint not found: {PRETRAINED_CONFIG['checkpoint_path']}")
            print("   You need to train a VQ-VAE first!")
            return
        
        # Create model from pretrained checkpoint
        print("🧠 Loading pretrained VQ-VAE...")
        model = create_behavioral_model_from_checkpoint(
            checkpoint_path=PRETRAINED_CONFIG['checkpoint_path'],
            model_class=PRETRAINED_CONFIG['model_class'],
            model_config=PRETRAINED_CONFIG,
            device=device,
            freeze_encoder=BEHAVIORAL_CONFIG['freeze_encoder'],
            freeze_codebook=BEHAVIORAL_CONFIG['freeze_codebook'],
            hidden_dim=BEHAVIORAL_CONFIG['hidden_dim'],
            dropout_rate=BEHAVIORAL_CONFIG['dropout_rate']
        )
        
        # Load velocity data
        print("\n📊 Loading velocity data...")
        train_loader, test_loader, info = create_velocity_dataloaders(
            session_ids=TRAINING_SESSION_IDS[:5],  # Start with 5 sessions
            batch_size=TRAIN_CONFIG['batch_size'],
            test_split=0.2,
            window_size=60,
            stride=30,
            num_neurons=1,  # Single neuron
            normalize=True
        )
        
        # Optimizer (only trainable parameters)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
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
                
                all_metrics = {**train_metrics, **test_metrics, 
                              'epoch': epoch, 'learning_rate': current_lr}
                wandb.log(all_metrics)
                
                print(f"Epoch {epoch:3d} | LR={current_lr:.6f}")
                print(f"  Loss: {test_metrics['test/loss']:.6f}")
                print(f"  🔥 R²={test_metrics['test/r2']:.4f}, Corr={test_metrics['test/correlation']:.4f}, SNR={test_metrics['test/snr']:.1f}dB")
                print(f"  📊 Var ratio={test_metrics['test/variance_ratio']:.3f}")
                
                if epoch % TRAIN_CONFIG['visualize_every'] == 0:
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
                    }, './results/velocity/best_velocity_transfer.pth')
                    print(f"  💾 Saved (corr={best_corr:.4f})")
                else:
                    patience_counter += 1
                
                if patience_counter >= TRAIN_CONFIG['patience']:
                    print("\n⚠️ Early stopping")
                    break
            else:
                wandb.log({**train_metrics, 'epoch': epoch})
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED")
        print("="*70)
        print(f"Best correlation: {best_corr:.4f}")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()