#!/usr/bin/env python3
"""
Training Script for Behavioral VQ-VAE

Uses pretrained neural encoder + codebook (frozen)
Trains only behavioral decoder to predict:
- Pupil dilation
- Vertical position  
- Horizontal position
- Velocity

Script di training che:

Carica modello pre-allenato
Allena SOLO decoder comportamentale
Logga metriche su W&B
Salva best model

Usage:
    python scripts/train_behavioral_vqvae.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wandb
import matplotlib.pyplot as plt
from datetime import datetime

from models.behavioral_vqvae import (
    create_behavioral_model_from_checkpoint,
    BehavioralVQVAE
)
from models.vqvae import CalciumVQVAE
from models.dual_vqvae import DualCalciumVQVAE

# ============================================================================
# CONFIGURATION
# ============================================================================

# Pretrained model configuration
PRETRAINED_CONFIG = {
    'checkpoint_path': 'results/best_single_neuron_model_32.pth',
    'model_class': CalciumVQVAE,  # or DualCalciumVQVAE
    
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

# Behavioral model configuration
BEHAVIORAL_CONFIG = {
    'freeze_encoder': True,
    'freeze_codebook': True,
    'hidden_dim': 256,  # Hidden layer size for behavioral decoder
    'dropout_rate': 0.3,
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'max_epochs': 100,
    'patience': 15,
    'weight_decay': 0.0001,
    
    # Loss weights (if you want to weight behavioral variables differently)
    'behavioral_weights': [1.0, 1.0, 1.0, 1.0],  # [pupil, vert, horiz, velocity]
    
    'visualize_every': 10,
    'evaluate_every': 5,
}

# ============================================================================
# DATA LOADING
# ============================================================================


USE_REAL_DATA = True  

def create_dataloaders(batch_size=32, train_split=0.8, use_real_data=USE_REAL_DATA):
    """
    Create train/test dataloaders
    
    Args:
        batch_size: Batch size
        train_split: Train/test split ratio
        use_real_data: If True, load real Allen Brain data. If False, use dummy data.
    """
    
    if use_real_data:
        # ====================================================================
        # REAL DATA from Allen Brain Observatory
        # ====================================================================
        print("\n📊 Loading REAL behavioral data from Allen Brain Observatory...")
        
        from datasets.behavioral import create_behavioral_dataloaders
        from datasets.calcium import TRAINING_SESSION_IDS
        
        # Use subset of sessions (or all)
        # For testing, start with 2-3 sessions
        # For full training, use all TRAINING_SESSION_IDS
        
        session_ids = TRAINING_SESSION_IDS[:3]  # Start with 3 sessions
        # session_ids = TRAINING_SESSION_IDS  # Uncomment for all sessions
        
        train_loader, test_loader, dataset_info = create_behavioral_dataloaders(
            session_ids=session_ids,
            batch_size=batch_size,
            test_split=train_split,
            window_size=60,
            stride=30,
            num_neurons=1,
            num_workers=0,
            normalize=True  # Z-score normalization
        )
        
        print(f"✅ Real data loaded:")
        print(f"   Sessions: {dataset_info['num_sessions']}")
        print(f"   Train samples: {dataset_info['train_samples']}")
        print(f"   Test samples: {dataset_info['test_samples']}")
        
    else:
        # ====================================================================
        # DUMMY DATA (for quick testing)
        # ====================================================================
        print("\n📊 Creating dataloaders...")
        print("⚠️  Using DUMMY data for testing!")
        print("   Set USE_REAL_DATA=True to use Allen Brain Observatory data")
        
        class DummyBehavioralDataset(Dataset):
            """Dummy dataset for testing"""
            def __init__(self, num_samples=1000, window_size=60, num_neurons=1):
                self.neural_data = torch.randn(num_samples, num_neurons, window_size)
                self.behavioral_data = torch.randn(num_samples, 4)
            
            def __len__(self):
                return len(self.neural_data)
            
            def __getitem__(self, idx):
                return self.neural_data[idx], self.behavioral_data[idx]
        
        full_dataset = DummyBehavioralDataset(num_samples=1000, window_size=60, num_neurons=1)
        
        train_size = int(len(full_dataset) * train_split)
        test_size = len(full_dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"✅ Dummy dataloaders created:")
        print(f"   Train: {train_size} samples")
        print(f"   Test: {test_size} samples")
    
    return train_loader, test_loader


# ============================================================================
# LOSS FUNCTION
# ============================================================================

def behavioral_loss(predictions, targets, weights=None):
    """
    Loss for behavioral prediction
    
    Args:
        predictions: (B, 4) - predicted behavioral variables
        targets: (B, 4) - ground truth
        weights: (4,) - optional weights for each variable
    
    Returns:
        total_loss: scalar
        individual_losses: dict with loss for each variable
    """
    
    if weights is None:
        weights = torch.ones(4, device=predictions.device)
    else:
        weights = torch.tensor(weights, device=predictions.device)
    
    # MSE for each variable
    mse_per_var = F.mse_loss(predictions, targets, reduction='none').mean(dim=0)
    
    # Weighted sum
    total_loss = (mse_per_var * weights).sum()
    
    individual_losses = {
        'pupil_dilation': mse_per_var[0].item(),
        'vertical_position': mse_per_var[1].item(),
        'horizontal_position': mse_per_var[2].item(),
        'velocity': mse_per_var[3].item(),
    }
    
    return total_loss, individual_losses


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, config):
    """Train one epoch"""
    model.train()
    
    # Only train behavioral decoder (encoder/codebook are frozen)
    model.behavioral_decoder.train()
    model.encoder.eval()
    model.vector_quantization.eval()
    
    total_loss = 0
    total_pupil_loss = 0
    total_vert_loss = 0
    total_horiz_loss = 0
    total_velocity_loss = 0
    
    for neural_data, behavioral_targets in dataloader:
        neural_data = neural_data.to(device)
        behavioral_targets = behavioral_targets.to(device)
        
        # Forward pass
        predictions = model(neural_data, return_quantized=False)
        
        # Compute loss
        loss, individual_losses = behavioral_loss(
            predictions, 
            behavioral_targets,
            weights=config['behavioral_weights']
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.behavioral_decoder.parameters(), 1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_pupil_loss += individual_losses['pupil_dilation']
        total_vert_loss += individual_losses['vertical_position']
        total_horiz_loss += individual_losses['horizontal_position']
        total_velocity_loss += individual_losses['velocity']
    
    num_batches = len(dataloader)
    
    return {
        'train/total_loss': total_loss / num_batches,
        'train/pupil_loss': total_pupil_loss / num_batches,
        'train/vert_loss': total_vert_loss / num_batches,
        'train/horiz_loss': total_horiz_loss / num_batches,
        'train/velocity_loss': total_velocity_loss / num_batches,
    }


def evaluate_model(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for neural_data, behavioral_targets in dataloader:
            neural_data = neural_data.to(device)
            behavioral_targets = behavioral_targets.to(device)
            
            predictions = model(neural_data, return_quantized=False)
            
            loss, _ = behavioral_loss(predictions, behavioral_targets)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(behavioral_targets.cpu().numpy())
            total_loss += loss.item()
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics per variable
    metrics = {}
    var_names = ['pupil', 'vert', 'horiz', 'velocity']
    
    for i, var_name in enumerate(var_names):
        pred_var = predictions[:, i]
        target_var = targets[:, i]
        
        # MSE
        mse = np.mean((pred_var - target_var) ** 2)
        
        # MAE
        mae = np.mean(np.abs(pred_var - target_var))
        
        # R²
        ss_res = np.sum((target_var - pred_var) ** 2)
        ss_tot = np.sum((target_var - np.mean(target_var)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        
        # Correlation
        if np.std(pred_var) > 1e-8 and np.std(target_var) > 1e-8:
            from scipy.stats import pearsonr
            corr, _ = pearsonr(pred_var, target_var)
            corr = corr if not np.isnan(corr) else 0.0
        else:
            corr = 0.0
        
        metrics[f'test/{var_name}_mse'] = mse
        metrics[f'test/{var_name}_mae'] = mae
        metrics[f'test/{var_name}_r2'] = r2
        metrics[f'test/{var_name}_corr'] = corr
    
    metrics['test/total_loss'] = total_loss / len(dataloader)
    
    return metrics


def visualize_predictions(model, dataloader, device, epoch):
    """Visualize predictions vs ground truth"""
    model.eval()
    
    # Get one batch
    neural_data, behavioral_targets = next(iter(dataloader))
    neural_data = neural_data.to(device)
    behavioral_targets = behavioral_targets.to(device)
    
    with torch.no_grad():
        predictions = model(neural_data, return_quantized=False)
    
    predictions = predictions.cpu().numpy()
    targets = behavioral_targets.cpu().numpy()
    
    # Plot first 4 samples
    num_samples = min(4, predictions.shape[0])
    
    fig, axes = plt.subplots(4, num_samples, figsize=(4*num_samples, 12))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    var_names = ['Pupil Dilation', 'Vertical Position', 'Horizontal Position', 'Velocity']
    
    for var_idx, var_name in enumerate(var_names):
        for sample_idx in range(num_samples):
            ax = axes[var_idx, sample_idx]
            
            target_val = targets[sample_idx, var_idx]
            pred_val = predictions[sample_idx, var_idx]
            error = abs(target_val - pred_val)
            
            # Bar plot
            ax.bar(['Target', 'Predicted'], [target_val, pred_val], 
                  color=['blue', 'red'], alpha=0.7)
            
            ax.set_title(f'Sample {sample_idx}: {var_name}\nError: {error:.3f}')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Behavioral Predictions - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    # Initialize W&B
    wandb.init(
        project="calcium-behavioral-vqvae",
        name=f"behavioral-transfer-{datetime.now().strftime('%m%d-%H%M')}",
        config={
            **PRETRAINED_CONFIG,
            **BEHAVIORAL_CONFIG,
            **TRAIN_CONFIG
        },
        tags=["behavioral", "transfer-learning", "frozen-encoder"]
    )
    
    print("="*70)
    print("🧠 BEHAVIORAL VQ-VAE TRAINING")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}\n")
    
    try:
        # Create behavioral model from pretrained checkpoint
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
        
        # Create dataloaders
        train_loader, test_loader = create_dataloaders(
            batch_size=TRAIN_CONFIG['batch_size'],
            train_split=0.8
        )
        
        # Optimizer (only for trainable parameters)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TRAIN_CONFIG['max_epochs'],
            eta_min=1e-6
        )
        
        # Training loop
        print("\n🚀 Starting training...")
        best_test_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(TRAIN_CONFIG['max_epochs']):
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, device, TRAIN_CONFIG)
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Evaluate
            if epoch % TRAIN_CONFIG['evaluate_every'] == 0:
                test_metrics = evaluate_model(model, test_loader, device)
                
                all_metrics = {**train_metrics, **test_metrics, 'epoch': epoch, 'learning_rate': current_lr}
                wandb.log(all_metrics)
                
                print(f"\nEpoch {epoch:3d}:")
                print(f"  Train Loss: {train_metrics['train/total_loss']:.6f}")
                print(f"  Test Loss: {test_metrics['test/total_loss']:.6f}")
                print(f"  Pupil R²: {test_metrics['test/pupil_r2']:.4f}, Corr: {test_metrics['test/pupil_corr']:.4f}")
                print(f"  Velocity R²: {test_metrics['test/velocity_r2']:.4f}, Corr: {test_metrics['test/velocity_corr']:.4f}")
                
                # Visualize
                if epoch % TRAIN_CONFIG['visualize_every'] == 0:
                    fig = visualize_predictions(model, test_loader, device, epoch)
                    wandb.log({f"predictions/epoch_{epoch}": wandb.Image(fig)})
                    plt.close(fig)
                
                # Save best model
                if test_metrics['test/total_loss'] < best_test_loss:
                    best_test_loss = test_metrics['test/total_loss']
                    patience_counter = 0
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_test_loss': best_test_loss,
                        'test_metrics': test_metrics
                    }
                    
                    os.makedirs('./results/behavioral', exist_ok=True)
                    torch.save(checkpoint, './results/behavioral/best_behavioral_model.pth')
                    print(f"  💾 Saved best model (loss={best_test_loss:.6f})")
                else:
                    patience_counter += 1
                
                if patience_counter >= TRAIN_CONFIG['patience']:
                    print(f"\n⚠️ Early stopping at epoch {epoch}")
                    break
            else:
                wandb.log({**train_metrics, 'epoch': epoch, 'learning_rate': current_lr})
        
        print("\n✅ Training completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()