#!/usr/bin/env python3
"""
Dual-Codebook VQ-VAE Training Script - 1024 Embeddings CON VINCOLO

Questo script addestra un modello dual codebook con 1024 embeddings totali
(512 active + 512 inactive) CON VINCOLO FORTE:
- Segnali inattivi → DEVONO usare codebook inactive
- Segnali attivi → DEVONO usare codebook active

Il vincolo è basato sull'attività REALE del segnale neurale (max deviation dalla baseline).

Uso:
    python scripts/train_dual_1024_constrained.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import wandb
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from models.dual_vqvae import DualCalciumVQVAE
from datasets.calcium import (
    create_unified_dataloaders,
    TRAINING_SESSION_IDS,
    TEST_SESSION_IDS
)

# ============================================================================
# MODEL CONFIGURATION - DUAL CODEBOOK 1024 EMBEDDINGS CON VINCOLO
# ============================================================================

MODEL_CONFIG = {
    # Architecture
    'num_neurons': 1,           # Single neuron
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    
    # 🔥 DUAL CODEBOOK: 1024 total = 512 active + 512 inactive
    'num_embeddings': 1024,     # ⬅️ 1024 embeddings totali
    'embedding_dim': 64,         # ⬅️ 64 dimensioni (più grande per più embeddings)
    'commitment_cost': 0.25,
    'dropout_rate': 0.1,
    'use_quantizer': True,
    
    # 🔥 DUAL CODEBOOK SPECIFIC - VINCOLO FORTE
    'threshold': 0.0,           # Sarà sovrascritto da adaptive
    'threshold_type': 'adaptive',  # Usa mediana per bilanciamento 50/50
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.0005,
    'max_epochs': 205,
    'patience': 40,
    'warmup_epochs': 30,
    'weight_decay': 0.0001,
    
    # Loss weights
    'temporal_smoothness_alpha': 0.05,
    
    # Visualization
    'visualize_every': 20,
    'evaluate_every': 5,
}

# ============================================================================
# FUNZIONI DI LOSS
# ============================================================================

def compute_loss_with_temporal_smoothness(recon, target, alpha=0.1):
    """Loss con smoothness temporale"""
    recon_loss = F.mse_loss(recon, target)
    
    # Smoothness: penalizza variazioni temporali brusche
    if recon.dim() == 3:  # (B, C, T)
        diff_recon = recon[:, :, 1:] - recon[:, :, :-1]
        diff_target = target[:, :, 1:] - target[:, :, :-1]
        smooth_loss = F.mse_loss(diff_recon, diff_target)
    else:
        smooth_loss = torch.tensor(0.0, device=recon.device)
    
    total_loss = recon_loss + alpha * smooth_loss
    
    return total_loss, recon_loss, smooth_loss

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch_dual_codebook(model, dataloader, optimizer, device, epoch, config):
    """Training con dual codebook VINCOLATO"""
    model.train()
    
    # VQ Weight Scheduling
    warmup_epochs = config['warmup_epochs']
    max_epochs = config['max_epochs']
    
    if epoch < warmup_epochs:
        vq_weight = 0.001 * (epoch / warmup_epochs)
        phase = "WARMUP"
    else:
        progress = (epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
        vq_weight = 0.001 + 0.1 * progress
        phase = "VQ_ACTIVE"
    
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    total_smooth_loss = 0
    
    for batch_idx, neural_data in enumerate(dataloader):
        neural_data = neural_data.to(device)
        
        # Forward pass
        # Il modello ora passa automaticamente x_original al quantizer
        vq_loss, recon, perplexity, _, _ = model(neural_data)
        
        # Custom loss with temporal smoothness
        combined_loss, recon_loss, smooth_loss = compute_loss_with_temporal_smoothness(
            recon, neural_data, alpha=config['temporal_smoothness_alpha']
        )
        
        # Total loss with VQ
        loss = combined_loss + vq_weight * vq_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Pulisci cache GPU periodicamente
        if device.type == 'cuda' and (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
        total_smooth_loss += smooth_loss if isinstance(smooth_loss, float) else smooth_loss.item()
    
    num_batches = len(dataloader)
    return {
        'train/total_loss': total_loss / num_batches,
        'train/recon_loss': total_recon_loss / num_batches,
        'train/vq_loss': total_vq_loss / num_batches,
        'train/perplexity': total_perplexity / num_batches,
        'train/smooth_loss': total_smooth_loss / num_batches,
        'train/vq_weight': vq_weight,
        'train/phase': phase
    }

def evaluate_dual_codebook(model, dataloader, device):
    """Valutazione con dual codebook"""
    model.eval()
    
    all_recon = []
    all_original = []
    
    with torch.no_grad():
        for neural_data in dataloader:
            neural_data = neural_data.to(device)
            
            vq_loss, recon, perplexity, _, _ = model(neural_data)
            
            all_original.append(neural_data.cpu().numpy())
            all_recon.append(recon.cpu().numpy())
    
    all_original = np.concatenate(all_original, axis=0)
    all_recon = np.concatenate(all_recon, axis=0)
    
    # Metriche
    mse = mean_squared_error(all_original.flatten(), all_recon.flatten())
    mae = mean_absolute_error(all_original.flatten(), all_recon.flatten())
    
    # Correlazione per neurone
    correlations = []
    for i in range(all_original.shape[0]):
        orig_flat = all_original[i].flatten()
        recon_flat = all_recon[i].flatten()
        if np.std(orig_flat) > 0 and np.std(recon_flat) > 0:
            corr, _ = pearsonr(orig_flat, recon_flat)
            correlations.append(corr)
    
    corr_mean = np.mean(correlations) if correlations else 0.0
    corr_std = np.std(correlations) if correlations else 0.0
    
    # R²
    ss_res = np.sum((all_original - all_recon) ** 2)
    ss_tot = np.sum((all_original - np.mean(all_original)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # SNR
    signal_power = np.mean(all_original ** 2)
    noise_power = np.mean((all_original - all_recon) ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0.0
    
    return {
        'test/mse': mse,
        'test/mae': mae,
        'test/correlation_mean': corr_mean,
        'test/correlation_std': corr_std,
        'test/r2': r2,
        'test/snr_db': snr_db,
        'test/perplexity': perplexity.item() if isinstance(perplexity, torch.Tensor) else perplexity
    }

def log_dual_codebook_stats(model, epoch):
    """Log statistics about dual codebook usage"""
    stats = model.get_codebook_usage()
    
    wandb.log({
        'codebook/total_usage': stats['total_usage_percentage'],
        'codebook/active_usage': stats['active_usage_percentage'],
        'codebook/inactive_usage': stats['inactive_usage_percentage'],
        'codebook/active_codes_used': stats['active_used_codes'],
        'codebook/inactive_codes_used': stats['inactive_used_codes'],
        'codebook/threshold': stats['threshold'],
    }, step=epoch)
    
    print(f"\n  📚 Dual Codebook Stats (Epoch {epoch}):")
    print(f"     Total usage: {stats['total_usage_percentage']:.1f}%")
    print(f"     Active: {stats['active_used_codes']}/{stats['active_total_codes']} ({stats['active_usage_percentage']:.1f}%)")
    print(f"     Inactive: {stats['inactive_used_codes']}/{stats['inactive_total_codes']} ({stats['inactive_usage_percentage']:.1f}%)")
    print(f"     Threshold: {stats['threshold']:.4f}")

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    # Initialize W&B
    wandb.init(
        project="calcium-vqvae-dual-codebook-constrained",
        name="dual-1024-constrained-single-neuron",
        config={
            "model_type": "DualCalciumVQVAE_Constrained_1024",
            "num_sessions_train": len(TRAINING_SESSION_IDS),
            "num_sessions_test": len(TEST_SESSION_IDS),
            "window_size": 60,
            "stride": 50,
            **MODEL_CONFIG,
            **TRAIN_CONFIG
        },
        tags=["dual-codebook", "single-neuron", "constrained", "1024-embeddings"]
    )
    
    print("🔥 DUAL-CODEBOOK VQ-VAE TRAINING (1024 Embeddings CON VINCOLO)")
    print("="*70)
    print("✅ Model config:")
    for key, value in MODEL_CONFIG.items():
        print(f"   {key}: {value}")
    print("\n✅ Training config:")
    for key, value in TRAIN_CONFIG.items():
        print(f"   {key}: {value}")
    print("="*70)
    
    # 🔧 CONFIGURAZIONE GPU/CPU
    FORCE_CPU = False  # ⬅️ Cambia a True se GPU da errori
    
    if FORCE_CPU:
        device = torch.device('cpu')
        print("⚠️  CPU FORZATO (FORCE_CPU=True)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.empty_cache()
            print(f"✅ GPU disponibile: {torch.cuda.get_device_name(0)}")
            print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  GPU non disponibile, usando CPU")
    
    print(f"💻 Using device: {device}")
    
    # Create results directory
    os.makedirs('./results/dual_codebook', exist_ok=True)
    
    try:
        # Load datasets
        print("\n📊 Loading datasets...")
        train_loader, test_loader, dataset_info = create_unified_dataloaders(
            train_session_ids=TRAINING_SESSION_IDS,
            batch_size=TRAIN_CONFIG['batch_size'],
            test_split=0.2,
            window_size=wandb.config.window_size,
            stride=wandb.config.stride,
            min_neurons=MODEL_CONFIG['num_neurons'],
            num_workers=0,
            use_cross_session_test=False
        )
        
        print(f"✅ Dataset loaded:")
        print(f"   Sessions: {dataset_info['num_sessions_train']}")
        print(f"   Train samples: {dataset_info['train_samples']}")
        print(f"   Test samples: {dataset_info['test_samples']}")
        
        # Create model
        print("\n🏗️  Creating dual codebook model (1024 embeddings, CON VINCOLO)...")
        model = DualCalciumVQVAE(
            num_neurons=MODEL_CONFIG['num_neurons'],
            num_hiddens=MODEL_CONFIG['num_hiddens'],
            num_residual_layers=MODEL_CONFIG['num_residual_layers'],
            num_residual_hiddens=MODEL_CONFIG['num_residual_hiddens'],
            num_embeddings=MODEL_CONFIG['num_embeddings'],
            embedding_dim=MODEL_CONFIG['embedding_dim'],
            commitment_cost=MODEL_CONFIG['commitment_cost'],
            dropout_rate=MODEL_CONFIG['dropout_rate'],
            use_quantizer=MODEL_CONFIG['use_quantizer'],
            threshold=MODEL_CONFIG['threshold'],
            threshold_type=MODEL_CONFIG['threshold_type']
        ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        wandb.log({"model/num_parameters": num_params})
        print(f"✅ Model created: {num_params:,} parameters")
        print(f"   Active codebook: {MODEL_CONFIG['num_embeddings']//2} embeddings")
        print(f"   Inactive codebook: {MODEL_CONFIG['num_embeddings']//2} embeddings")
        print(f"   ⚠️  VINCOLO ATTIVO: codebook selezionato in base ad attività reale del segnale")
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=TRAIN_CONFIG['max_epochs'],
            eta_min=1e-6
        )
        
        # Training loop
        print("\n🚀 Starting training...")
        best_test_corr = -1
        patience_counter = 0
        
        for epoch in range(TRAIN_CONFIG['max_epochs']):
            # Training
            train_metrics = train_epoch_dual_codebook(
                model, train_loader, optimizer, device, epoch, TRAIN_CONFIG
            )
            
            # Log codebook stats
            if epoch % TRAIN_CONFIG['evaluate_every'] == 0:
                log_dual_codebook_stats(model, epoch)
            
            # Evaluation
            if epoch % TRAIN_CONFIG['evaluate_every'] == 0:
                test_metrics = evaluate_dual_codebook(model, test_loader, device)
                
                # Log metrics
                wandb.log({**train_metrics, **test_metrics}, step=epoch)
                
                # Print
                print(f"\nEpoch {epoch:4d} [{train_metrics['train/phase']}]:")
                print(f"  Train Loss: {train_metrics['train/total_loss']:.6f}")
                print(f"  Test MSE: {test_metrics['test/mse']:.6f}")
                print(f"  Test Corr: {test_metrics['test/correlation_mean']:.4f} ± {test_metrics['test/correlation_std']:.4f}")
                print(f"  R²: {test_metrics['test/r2']:.4f}, SNR: {test_metrics['test/snr_db']:.1f} dB")
                print(f"  Perplexity: {test_metrics['test/perplexity']:.1f}")
                
                # Save best model
                if test_metrics['test/correlation_mean'] > best_test_corr:
                    best_test_corr = test_metrics['test/correlation_mean']
                    checkpoint_path = './results/dual_codebook/best_dual_model_1024_constrained.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'test_correlation': best_test_corr,
                        'model_config': MODEL_CONFIG,
                    }, checkpoint_path)
                    print(f"  💾 Saved best model (corr={best_test_corr:.4f})")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= TRAIN_CONFIG['patience']:
                    print(f"\n⏹️  Early stopping at epoch {epoch}")
                    break
            
            scheduler.step()
        
        # Final stats
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETATO!")
        print("="*70)
        final_codebook_stats = model.get_codebook_usage()
        print(f"  Total usage: {final_codebook_stats['total_usage_percentage']:.1f}%")
        print(f"  Active usage: {final_codebook_stats['active_usage_percentage']:.1f}%")
        print(f"  Inactive usage: {final_codebook_stats['inactive_usage_percentage']:.1f}%")
        print(f"  Best correlation: {best_test_corr:.4f}")
        
        wandb.log({
            'final/codebook_total_usage': final_codebook_stats['total_usage_percentage'],
            'final/codebook_active_usage': final_codebook_stats['active_usage_percentage'],
            'final/codebook_inactive_usage': final_codebook_stats['inactive_usage_percentage'],
            'final/best_correlation': best_test_corr,
        })
        
    except Exception as e:
        print(f"\n❌ Errore durante il training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        wandb.finish()
        print("✅ W&B run completato!")


if __name__ == "__main__":
    main()

