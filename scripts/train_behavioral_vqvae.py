#!/usr/bin/env python3
"""
Training Script for Behavioral VQ-VAE - SESSION-BY-SESSION

Strategy:
1. Train on ONE session at a time
2. Use ALL neurons from that session
3. Batch size = number of neurons
4. Decoder aggregates information from all neurons
5. Output: 1 velocity per session
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
from datasets.velocity import create_single_session_dataloaders
from datasets.calcium import TRAINING_SESSION_IDS

# ============================================================================
# CONFIGURATION
# ============================================================================

PRETRAINED_CONFIG = {
    'checkpoint_path': 'results/best_single_neuron_model_32.pth',
    'model_class': CalciumVQVAE,
    
    'num_neurons': 1,  # Encoder trained on single neurons
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
    'dropout_rate': 0.1,
}

TRAIN_CONFIG = {
    'learning_rate': 0.01,
    'max_epochs': 200,
    'patience': 30,
    'weight_decay': 0.0001,
    'visualize_every': 10,
    'evaluate_every': 5,
    'gradient_clip': 1.0,
}

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Train one epoch"""
    model.train()
    model.velocity_decoder.train()
    model.encoder.eval()
    model.vector_quantization.eval()
    
    total_loss = 0
    first_batch = True
    
    for neural_data, velocity_target in dataloader:
        neural_data = neural_data.squeeze(0).to(device)
        neural_data = neural_data.unsqueeze(1)
        velocity_target = velocity_target.to(device)
        
        # Forward pass
        prediction = model(neural_data, return_quantized=False)
        
        # ========================================================================
        # 🔍 DEBUG: Print first batch info
        # ========================================================================
        if first_batch:
            print(f"\n🔍 First Batch Debug:")
            print(f"   Neural data shape: {neural_data.shape}")
            print(f"   Prediction: {prediction.item():.6f}")
            print(f"   Target: {velocity_target.item():.6f}")
            print(f"   Prediction requires_grad: {prediction.requires_grad}")
        # ========================================================================
        
        # Loss
        loss = F.smooth_l1_loss(prediction.unsqueeze(0), velocity_target)
        
        # ========================================================================
        # 🔍 DEBUG: Check gradients
        # ========================================================================
        if first_batch:
            print(f"   Loss: {loss.item():.6f}")
            print(f"   Loss requires_grad: {loss.requires_grad}")
        # ========================================================================
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # ========================================================================
        # 🔍 DEBUG: Check if gradients exist
        # ========================================================================
        if first_batch:
            has_grad = False
            total_grad_norm = 0
            for name, param in model.velocity_decoder.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    if not has_grad:
                        print(f"\n   ✅ Gradients found in decoder:")
                        has_grad = True
                    print(f"      {name}: grad_norm={grad_norm:.6f}")
                    if len([n for n, p in model.velocity_decoder.named_parameters()]) > 5:
                        break  # Print only first few
            
            if not has_grad:
                print(f"\n   ❌ NO GRADIENTS IN DECODER!")
            else:
                print(f"   Total gradient norm: {total_grad_norm:.6f}")
            
            first_batch = False
        # ========================================================================
        
        torch.nn.utils.clip_grad_norm_(model.velocity_decoder.parameters(), 0.5)
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
        for neural_data, velocity_target in dataloader:
            neural_data = neural_data.squeeze(0).to(device)
            neural_data = neural_data.unsqueeze(1)
            velocity_target = velocity_target.to(device)
            
            prediction = model(neural_data, return_quantized=False)
            loss = F.smooth_l1_loss(prediction.unsqueeze(0), velocity_target)
            
            all_predictions.append(prediction.cpu().item())
            all_targets.append(velocity_target.cpu().item())
            total_loss += loss.item()
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # ========================================================================
    # 🔍 DEBUG: Verifica predictions
    # ========================================================================
    print(f"\n🔍 DEBUG Predictions vs Targets:")
    print(f"   Targets    - Min: {targets.min():.4f}, Max: {targets.max():.4f}, Mean: {targets.mean():.4f}, Std: {targets.std():.4f}")
    print(f"   Predictions- Min: {predictions.min():.4f}, Max: {predictions.max():.4f}, Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")
    print(f"   First 20 predictions: {predictions[:20]}")
    print(f"   First 20 targets:     {targets[:20]}")
    
    if predictions.std() < 1e-6:
        print(f"   ⚠️ WARNING: Predictions are CONSTANT! Value: {predictions[0]:.6f}")
    # ========================================================================
    
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
    
    signal_power = np.mean(targets ** 2)
    noise_power = np.mean((targets - predictions) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
    
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
    """Visualize predictions"""
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for neural_data, velocity_target in dataloader:
            neural_data = neural_data.squeeze(0).to(device)
            neural_data = neural_data.unsqueeze(1)
            
            pred = model(neural_data).cpu().item()
            target = velocity_target.cpu().item()
            
            predictions.append(pred)
            targets.append(target)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter
    axes[0].scatter(targets, predictions, alpha=0.5, s=20)
    axes[0].plot([targets.min(), targets.max()], 
                [targets.min(), targets.max()], 'r--', lw=2)
    axes[0].set_xlabel('Target Velocity')
    axes[0].set_ylabel('Predicted Velocity')
    axes[0].set_title(f'Epoch {epoch}')
    axes[0].grid(True, alpha=0.3)
    
    if np.std(predictions) > 1e-8:
        corr, _ = pearsonr(predictions, targets)
        axes[0].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                    transform=axes[0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Histogram
    axes[1].hist(targets, bins=30, alpha=0.5, label='Target', density=True)
    axes[1].hist(predictions, bins=30, alpha=0.5, label='Pred', density=True)
    axes[1].set_xlabel('Velocity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN - TRAIN ON SINGLE SESSION
# ============================================================================

def train_single_session(session_id):
    """Train model on a single session"""
    
    wandb.init(
        project="calcium-velocity-session",
        name=f"session-{session_id}-{datetime.now().strftime('%m%d-%H%M')}",
        config={
            **PRETRAINED_CONFIG,
            **BEHAVIORAL_CONFIG,
            **TRAIN_CONFIG,
            'session_id': session_id
        },
        tags=["single-session", "all-neurons", "velocity"]
    )
    
    print("="*70)
    print(f"🚀 VELOCITY PREDICTION - SESSION {session_id}")
    print("="*70)
    print("📊 Strategy:")
    print("   1. Use ALL neurons from this session")
    print("   2. Aggregate neuron information in decoder")
    print("   3. Predict session-level velocity")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}\n")
    
    try:
        # Check checkpoint
        if not os.path.exists(PRETRAINED_CONFIG['checkpoint_path']):
            print(f"❌ Checkpoint not found!")
            return
        
        # Create model
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
        
        # Load session data
        print(f"\n📊 Loading data for session {session_id}...")
        train_loader, test_loader, info = create_single_session_dataloaders(
            session_id=session_id,
            test_split=0.2,
            window_size=60,
            stride=30,
            normalize=True
        )
        
        # ========================================================================
        # ✅ VERIFICA CLIPPING NEI DATI CARICATI
        # ========================================================================
        print(f"\n🔍 VERIFYING DATA CLIPPING...")
        
        # Collect all targets from train loader
        all_targets = []
        for _, velocity_target in train_loader:
            all_targets.append(velocity_target.item())
        
        all_targets = np.array(all_targets)
        
        print(f"   Loaded targets statistics:")
        print(f"   Min: {all_targets.min():.4f}, Max: {all_targets.max():.4f}")
        print(f"   Mean: {all_targets.mean():.4f}, Std: {all_targets.std():.4f}")
        
        # Check if clipping worked
        if np.abs(all_targets).max() > 10.0:
            print(f"   ❌ ERROR: Outliers detected! Max abs value: {np.abs(all_targets).max():.4f}")
            print(f"   ⚠️ Clipping DID NOT work! Check datasets/velocity.py")
            print(f"   First 20 targets: {all_targets[:20]}")
            
            # Continue anyway but warn
            print(f"\n   ⚠️ Continuing training but results may be poor...")
        else:
            print(f"   ✅ Clipping successful! Data range within ±3σ")
        
        # Recreate loaders (iterator was consumed)
        train_loader, test_loader, info = create_single_session_dataloaders(
            session_id=session_id,
            test_split=0.2,
            window_size=60,
            stride=30,
            normalize=True
        )
        # ========================================================================
        
        print(f"\n📊 Session info:")
        print(f"   Neurons: {info['num_neurons']}")
        print(f"   Train windows: {info['train_samples']}")
        print(f"   Test windows: {info['test_samples']}")

        # ========================================================================
        # BUILD DECODER BEFORE CREATING OPTIMIZER
        # ========================================================================
        print("\n🔨 Building decoder before creating optimizer...")

        # Get first batch to determine decoder size
        first_batch = next(iter(train_loader))
        neural_data_sample, _ = first_batch

        # Remove batch dim and add channel
        neural_data_sample = neural_data_sample.squeeze(0).to(device)  # (num_neurons, 60)
        neural_data_sample = neural_data_sample.unsqueeze(1)  # (num_neurons, 1, 60)

        # Run one forward pass to build decoder
        with torch.no_grad():
            _ = model(neural_data_sample, return_quantized=False)

        print("✅ Decoder built successfully!")
        
        # Recreate loader again (iterator consumed)
        train_loader, test_loader, info = create_single_session_dataloaders(
            session_id=session_id,
            test_split=0.2,
            window_size=60,
            stride=30,
            normalize=True
        )

        # ========================================================================
        # CREATE OPTIMIZER
        # ========================================================================
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay'],
            amsgrad=True
        )

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Optimizer created with {trainable_params:,} trainable params")
        
        # ========================================================================
        # SCHEDULER
        # ========================================================================
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=20,  # Restart ogni 20 epochs
            T_mult=1,
            eta_min=1e-6,  # LR minimo
            verbose=True
        )
        
        print("\n🚀 Starting training...\n")
        
        # ========================================================================
        # TRAINING LOOP
        # ========================================================================
        best_corr = -1.0
        patience_counter = 0
        
        for epoch in range(TRAIN_CONFIG['max_epochs']):
            train_metrics = train_epoch(model, train_loader, optimizer, device)
            
            if epoch % TRAIN_CONFIG['evaluate_every'] == 0:
                test_metrics = evaluate_model(model, test_loader, device)
                
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                
                all_metrics = {**train_metrics, **test_metrics, 
                              'epoch': epoch, 'learning_rate': current_lr}
                wandb.log(all_metrics)
                
                print(f"Epoch {epoch:3d} | LR={current_lr:.6f}")
                print(f"  Loss: {test_metrics['test/loss']:.6f}")
                print(f"  🔥 R²={test_metrics['test/r2']:.4f}, Corr={test_metrics['test/correlation']:.4f}")
                
                # Visualize predictions
                if epoch % TRAIN_CONFIG['visualize_every'] == 0:
                    fig = visualize_predictions(model, test_loader, device, epoch)
                    wandb.log({f"predictions/epoch_{epoch}": wandb.Image(fig)})
                    plt.close(fig)
                
                # Save best model
                if test_metrics['test/correlation'] > best_corr:
                    best_corr = test_metrics['test/correlation']
                    patience_counter = 0
                    
                    os.makedirs(f'./results/session_{session_id}', exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'session_id': session_id,
                        'num_neurons': info['num_neurons'],
                        'model_state_dict': model.state_dict(),
                        'best_correlation': best_corr,
                        'test_metrics': test_metrics,
                    }, f'./results/session_{session_id}/best_model.pth')
                    print(f"  💾 Saved (corr={best_corr:.4f})")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= TRAIN_CONFIG['patience']:
                    print("\n⚠️ Early stopping")
                    break
            else:
                wandb.log({**train_metrics, 'epoch': epoch})
        
        # ========================================================================
        # TRAINING COMPLETED
        # ========================================================================
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED")
        print("="*70)
        print(f"Best correlation: {best_corr:.4f}")
        print("="*70)
        
        wandb.finish()
        
        return best_corr
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        return None


# ============================================================================
# MAIN - TRAIN ALL SESSIONS
# ============================================================================

def main():
    """Train on ONE specific session"""
    
    # ========================================================================
    # ✅ SCEGLI LA SESSIONE QUI
    # ========================================================================
    
    SESSION_INDEX = 0  # ⚠️ CAMBIA QUESTO: 0-9 per le 10 sessioni
    
    # Oppure usa direttamente l'ID:
    # SESSION_ID = 650389887  # ⚠️ Decommenta e usa questo se preferisci l'ID esplicito
    
    # ========================================================================
    
    # Get session_id from index
    if 'SESSION_ID' not in locals():
        if SESSION_INDEX >= len(TRAINING_SESSION_IDS):
            print(f"❌ Invalid SESSION_INDEX: {SESSION_INDEX}")
            print(f"   Valid range: 0-{len(TRAINING_SESSION_IDS)-1}")
            print("\nAvailable sessions:")
            for i, sid in enumerate(TRAINING_SESSION_IDS):
                print(f"   [{i}] {sid}")
            return
        
        SESSION_ID = TRAINING_SESSION_IDS[SESSION_INDEX]
    
    # Print session info
    print("="*70)
    print("🚀 TRAINING ON SINGLE SESSION")
    print("="*70)
    print(f"   Session Index: {SESSION_INDEX}")
    print(f"   Session ID: {SESSION_ID}")
    print("="*70)
    
    # Train
    best_corr = train_single_session(SESSION_ID)
    
    # Final result
    print("\n" + "="*70)
    print("📊 FINAL RESULT")
    print("="*70)
    print(f"Session {SESSION_ID}: {best_corr:.4f if best_corr else 'FAILED'}")
    print("="*70)


if __name__ == "__main__":
    main()