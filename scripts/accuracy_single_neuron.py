#!/usr/bin/env python3
"""
Analisi dell'impatto della dimensione del codebook sull'accuratezza.
MODALITÀ: SOLO SINGOLO NEURONE

Testa diverse configurazioni (8, 16, 32, 64, 128, 256, 512, 1024, 2048 codici)
per vedere come varia la qualità della ricostruzione del singolo neurone.

Risultati salvati SOLO su W&B (no file locali).

Uso:
    python scripts/analyze_codebook_size_impact.py
    python scripts/analyze_codebook_size_impact.py --sizes 8 16 32 64 128
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
from pathlib import Path
import argparse
from scipy.stats import pearsonr
from tqdm import tqdm

from models.vqvae import CalciumVQVAE
from datasets.calcium import (
    create_unified_dataloaders,
    UnifiedMultiSessionDataset,
    TRAINING_SESSION_IDS,
    TEST_SESSION_IDS
)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# Configurazioni di codebook da testare
CODEBOOK_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Configurazione SINGOLO NEURONE
MODEL_CONFIG = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'embedding_dim': 32,
    'commitment_cost': 0.25,
    'dropout_rate': 0.1,
    'use_quantizer': True,
}

# Training params
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.0005,
    'max_epochs': 100,
    'patience': 20,
    'window_size': 60,
    'stride': 50,
    'warmup_epochs': 10,
}

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_model(num_embeddings):
    """Crea modello single neuron con numero di embeddings specificato"""
    return CalciumVQVAE(
        num_neurons=MODEL_CONFIG['num_neurons'],
        num_hiddens=MODEL_CONFIG['num_hiddens'],
        num_residual_layers=MODEL_CONFIG['num_residual_layers'],
        num_residual_hiddens=MODEL_CONFIG['num_residual_hiddens'],
        num_embeddings=num_embeddings,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        commitment_cost=MODEL_CONFIG['commitment_cost'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        use_quantizer=MODEL_CONFIG['use_quantizer']
    )


def train_epoch(model, dataloader, optimizer, device, epoch, max_epochs):
    """Training epoch con VQ scheduling"""
    model.train()
    
    # VQ weight scheduling
    warmup_epochs = TRAINING_CONFIG['warmup_epochs']
    if epoch < warmup_epochs:
        progress = epoch / warmup_epochs
        vq_weight = 0.0 + progress * 0.1
    else:
        progress = (epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
        vq_weight = 0.1 + progress * 0.2
        vq_weight = min(vq_weight, 0.3)
    
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    
    for neural_data in dataloader:
        neural_data = neural_data.to(device)
        
        # Forward pass
        vq_loss, recon, perplexity, _, _ = model(neural_data)
        
        # Huber loss
        diff = torch.abs(recon - neural_data)
        delta = 1.0
        recon_loss = torch.where(
            diff < delta,
            0.5 * diff ** 2,
            delta * (diff - 0.5 * delta)
        ).mean()
        
        # Combined loss
        loss = recon_loss + vq_weight * vq_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
        'perplexity': total_perplexity / num_batches,
        'vq_weight': vq_weight
    }


def evaluate_model(model, dataloader, device):
    """Valutazione per singolo neurone"""
    model.eval()
    all_original = []
    all_reconstructed = []
    total_vq_loss = 0
    total_perplexity = 0
    num_batches = 0
    
    with torch.no_grad():
        for neural_data in dataloader:
            neural_data = neural_data.to(device)
            vq_loss, recon, perplexity, _, _ = model(neural_data)
            
            all_original.append(neural_data.cpu().numpy())
            all_reconstructed.append(recon.cpu().numpy())
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
    
    original = np.concatenate(all_original, axis=0)
    reconstructed = np.concatenate(all_reconstructed, axis=0)
    
    # MSE e MAE
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    
    # R²
    y_true = original.flatten()
    y_pred = reconstructed.flatten()
    y_mean = np.mean(y_true)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    
    # Correlazioni temporali (singolo neurone)
    correlations = []
    for sample_idx in range(original.shape[0]):
        orig_trace = original[sample_idx, 0, :]
        recon_trace = reconstructed[sample_idx, 0, :]
        
        if np.std(orig_trace) > 1e-8 and np.std(recon_trace) > 1e-8:
            corr, _ = pearsonr(orig_trace, recon_trace)
            correlations.append(corr if not np.isnan(corr) else 0)
    
    # Signal-to-noise ratio
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
    
    return {
        'mse': mse,
        'mae': mae,
        'r_squared': r_squared,
        'correlation_mean': np.mean(correlations) if correlations else 0,
        'correlation_std': np.std(correlations) if correlations else 0,
        'correlation_median': np.median(correlations) if correlations else 0,
        'correlation_min': np.min(correlations) if correlations else 0,
        'correlation_max': np.max(correlations) if correlations else 1,
        'snr_db': snr,
        'vq_loss': total_vq_loss / num_batches if num_batches > 0 else 0,
        'perplexity': total_perplexity / num_batches if num_batches > 0 else 0,
        'num_samples': original.shape[0],
        'all_correlations': correlations
    }


def train_and_evaluate_codebook_size(num_embeddings, device='cuda'):
    """
    Train e valuta modello single neuron con numero specifico di embeddings.
    
    Args:
        num_embeddings: Numero di codici nel codebook
        device: torch device
    
    Returns:
        dict con risultati di training e test
    """
    
    print(f"\n{'='*70}")
    print(f"🔬 Testing {num_embeddings} codici (SINGLE NEURON)")
    print(f"{'='*70}")
    
    # Dataset
    print(f"📊 Loading dataset...")
    train_loader, test_loader, dataset_info = create_unified_dataloaders(
        train_session_ids=TRAINING_SESSION_IDS,
        batch_size=TRAINING_CONFIG['batch_size'],
        test_split=0.2,
        window_size=TRAINING_CONFIG['window_size'],
        stride=TRAINING_CONFIG['stride'],
        min_neurons=MODEL_CONFIG['num_neurons'],  # 1 neuron
        num_workers=0,
        use_cross_session_test=False
    )
    
    print(f"✅ Dataset: train={dataset_info['train_samples']}, test={dataset_info['test_samples']}")
    
    # Model
    print(f"🏗️ Creating model...")
    model = create_model(num_embeddings).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model: {num_params:,} parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=0.0001,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=TRAINING_CONFIG['max_epochs'],
        eta_min=1e-6
    )
    
    # Training loop
    print(f"🚀 Training...")
    best_test_mse = float('inf')
    best_train_loss = float('inf')
    patience_counter = 0
    
    training_history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_perplexity': [],
        'test_mse': [],
        'test_correlation': [],
        'test_r_squared': []
    }
    
    for epoch in tqdm(range(TRAINING_CONFIG['max_epochs']), desc=f"Training {num_embeddings} codes"):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, 
            epoch, TRAINING_CONFIG['max_epochs']
        )
        
        scheduler.step()
        
        # Save history
        training_history['train_loss'].append(train_metrics['loss'])
        training_history['train_recon_loss'].append(train_metrics['recon_loss'])
        training_history['train_perplexity'].append(train_metrics['perplexity'])
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            test_metrics = evaluate_model(model, test_loader, device)
            
            training_history['test_mse'].append(test_metrics['mse'])
            training_history['test_correlation'].append(test_metrics['correlation_mean'])
            training_history['test_r_squared'].append(test_metrics['r_squared'])
            
            current_test_mse = test_metrics['mse']
            current_train_loss = train_metrics['recon_loss']
            
            # Early stopping
            if current_train_loss < best_train_loss:
                best_train_loss = current_train_loss
                best_test_mse = current_test_mse
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= TRAINING_CONFIG['patience']:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    # Final evaluation
    print(f"📊 Final evaluation...")
    final_test_metrics = evaluate_model(model, test_loader, device)
    final_train_metrics = train_epoch(model, train_loader, optimizer, device, 
                                     epoch if 'epoch' in locals() else TRAINING_CONFIG['max_epochs'] - 1,
                                     TRAINING_CONFIG['max_epochs'])
    
    print(f"✅ Results:")
    print(f"   Train Loss: {final_train_metrics['recon_loss']:.6f}")
    print(f"   Test MSE: {final_test_metrics['mse']:.6f}")
    print(f"   Test Corr: {final_test_metrics['correlation_mean']:.4f}")
    print(f"   R²: {final_test_metrics['r_squared']:.4f}")
    print(f"   Perplexity: {final_train_metrics['perplexity']:.1f}")
    
    # Codebook usage
    if hasattr(model, 'get_codebook_usage'):
        usage_stats = model.get_codebook_usage()
        usage_pct = usage_stats.get('usage_percentage', 0)
        print(f"   Codebook usage: {usage_pct:.1f}%")
    else:
        usage_pct = 100.0
    
    return {
        'num_embeddings': num_embeddings,
        'num_parameters': num_params,
        'final_train_loss': final_train_metrics['recon_loss'],
        'final_test_metrics': final_test_metrics,
        'training_history': training_history,
        'codebook_usage_pct': usage_pct,
        'best_train_loss': best_train_loss,
        'best_test_mse': best_test_mse,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_codebook_analysis(results):
    """Crea grafici completi e li logga SOLO su W&B (no file locali)"""
    
    # Estrai metriche
    codebook_sizes = [r['num_embeddings'] for r in results]
    train_losses = [r['final_train_loss'] for r in results]
    test_mses = [r['final_test_metrics']['mse'] for r in results]
    test_corrs = [r['final_test_metrics']['correlation_mean'] for r in results]
    test_r2 = [r['final_test_metrics']['r_squared'] for r in results]
    perplexities = [r['final_test_metrics']['perplexity'] for r in results]
    codebook_usage = [r['codebook_usage_pct'] for r in results]
    num_params = [r['num_parameters'] for r in results]
    
    # ========================================================================
    # FIGURA PRINCIPALE: 3x3 grid
    # ========================================================================
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    title = f"📊 Codebook Size Impact Analysis - Single Neuron"
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # === ROW 1: Metriche vs Codebook Size ===
    
    # 1. Train Loss
    axes[0, 0].plot(codebook_sizes, train_losses, 'b-o', linewidth=2, markersize=8)
    axes[0, 0].set_xscale('log', base=2)
    axes[0, 0].set_xlabel('Codebook Size', fontsize=11)
    axes[0, 0].set_ylabel('Train Loss (MSE)', fontsize=11)
    axes[0, 0].set_title('Training Loss vs Codebook Size', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(codebook_sizes)
    axes[0, 0].set_xticklabels([str(s) for s in codebook_sizes], rotation=45)
    
    best_train_idx = np.argmin(train_losses)
    axes[0, 0].scatter(codebook_sizes[best_train_idx], train_losses[best_train_idx], 
                       color='red', s=200, marker='*', zorder=5, label='Best')
    axes[0, 0].legend()
    
    # 2. Test MSE
    axes[0, 1].plot(codebook_sizes, test_mses, 'r-o', linewidth=2, markersize=8)
    axes[0, 1].set_xscale('log', base=2)
    axes[0, 1].set_xlabel('Codebook Size', fontsize=11)
    axes[0, 1].set_ylabel('Test MSE', fontsize=11)
    axes[0, 1].set_title('Test MSE vs Codebook Size', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(codebook_sizes)
    axes[0, 1].set_xticklabels([str(s) for s in codebook_sizes], rotation=45)
    
    best_mse_idx = np.argmin(test_mses)
    axes[0, 1].scatter(codebook_sizes[best_mse_idx], test_mses[best_mse_idx], 
                       color='darkred', s=200, marker='*', zorder=5, label='Best')
    axes[0, 1].legend()
    
    # 3. Test Correlation
    axes[0, 2].plot(codebook_sizes, test_corrs, 'g-o', linewidth=2, markersize=8)
    axes[0, 2].set_xscale('log', base=2)
    axes[0, 2].set_xlabel('Codebook Size', fontsize=11)
    axes[0, 2].set_ylabel('Correlation', fontsize=11)
    axes[0, 2].set_title('Test Correlation vs Codebook Size', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xticks(codebook_sizes)
    axes[0, 2].set_xticklabels([str(s) for s in codebook_sizes], rotation=45)
    axes[0, 2].set_ylim([0.5, 1.0])
    
    best_corr_idx = np.argmax(test_corrs)
    axes[0, 2].scatter(codebook_sizes[best_corr_idx], test_corrs[best_corr_idx], 
                       color='darkgreen', s=200, marker='*', zorder=5, label='Best')
    axes[0, 2].legend()
    
    # === ROW 2: R², Perplexity, Codebook Usage ===
    
    # 4. R² Score
    axes[1, 0].plot(codebook_sizes, test_r2, 'purple', marker='o', linewidth=2, markersize=8)
    axes[1, 0].set_xscale('log', base=2)
    axes[1, 0].set_xlabel('Codebook Size', fontsize=11)
    axes[1, 0].set_ylabel('R² Score', fontsize=11)
    axes[1, 0].set_title('R² Score vs Codebook Size', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(codebook_sizes)
    axes[1, 0].set_xticklabels([str(s) for s in codebook_sizes], rotation=45)
    axes[1, 0].set_ylim([0, 1.0])
    
    best_r2_idx = np.argmax(test_r2)
    axes[1, 0].scatter(codebook_sizes[best_r2_idx], test_r2[best_r2_idx], 
                       color='darkviolet', s=200, marker='*', zorder=5, label='Best')
    axes[1, 0].legend()
    
    # 5. Perplexity
    axes[1, 1].plot(codebook_sizes, perplexities, 'orange', marker='o', linewidth=2, markersize=8)
    axes[1, 1].set_xscale('log', base=2)
    axes[1, 1].set_xlabel('Codebook Size', fontsize=11)
    axes[1, 1].set_ylabel('Perplexity', fontsize=11)
    axes[1, 1].set_title('Perplexity vs Codebook Size', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(codebook_sizes)
    axes[1, 1].set_xticklabels([str(s) for s in codebook_sizes], rotation=45)
    axes[1, 1].plot(codebook_sizes, codebook_sizes, 'k--', alpha=0.3, label='Ideal')
    axes[1, 1].legend()
    
    # 6. Codebook Usage
    axes[1, 2].plot(codebook_sizes, codebook_usage, 'cyan', marker='o', linewidth=2, markersize=8)
    axes[1, 2].set_xscale('log', base=2)
    axes[1, 2].set_xlabel('Codebook Size', fontsize=11)
    axes[1, 2].set_ylabel('Usage (%)', fontsize=11)
    axes[1, 2].set_title('Codebook Usage vs Size', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xticks(codebook_sizes)
    axes[1, 2].set_xticklabels([str(s) for s in codebook_sizes], rotation=45)
    axes[1, 2].set_ylim([0, 105])
    axes[1, 2].axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100%')
    axes[1, 2].legend()
    
    # === ROW 3: Trade-offs ===
    
    # 7. Accuracy vs Complexity
    axes[2, 0].scatter(num_params, test_corrs, c=codebook_sizes, 
                       cmap='viridis', s=200, alpha=0.7, edgecolors='black')
    axes[2, 0].set_xlabel('Number of Parameters', fontsize=11)
    axes[2, 0].set_ylabel('Test Correlation', fontsize=11)
    axes[2, 0].set_title('Accuracy vs Model Complexity', fontsize=12, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    for i, size in enumerate(codebook_sizes):
        axes[2, 0].annotate(str(size), (num_params[i], test_corrs[i]), 
                           fontsize=8, ha='center', va='bottom')
    
    # 8. Correlation vs MSE
    axes[2, 1].scatter(test_mses, test_corrs, c=codebook_sizes, 
                       cmap='plasma', s=200, alpha=0.7, edgecolors='black')
    axes[2, 1].set_xlabel('Test MSE', fontsize=11)
    axes[2, 1].set_ylabel('Test Correlation', fontsize=11)
    axes[2, 1].set_title('Correlation vs MSE Trade-off', fontsize=12, fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    for i, size in enumerate(codebook_sizes):
        axes[2, 1].annotate(str(size), (test_mses[i], test_corrs[i]), 
                           fontsize=8, ha='center', va='bottom')
    
    # 9. Summary Table
    axes[2, 2].axis('off')
    
    best_corr_size = codebook_sizes[best_corr_idx]
    best_mse_size = codebook_sizes[best_mse_idx]
    best_r2_size = codebook_sizes[best_r2_idx]
    
    summary_text = "🏆 BEST CONFIGURATIONS\n\n"
    summary_text += f"Best Correlation:\n"
    summary_text += f"  {best_corr_size} codes\n"
    summary_text += f"  Corr: {test_corrs[best_corr_idx]:.4f}\n\n"
    summary_text += f"Best MSE:\n"
    summary_text += f"  {best_mse_size} codes\n"
    summary_text += f"  MSE: {test_mses[best_mse_idx]:.6f}\n\n"
    summary_text += f"Best R²:\n"
    summary_text += f"  {best_r2_size} codes\n"
    summary_text += f"  R²: {test_r2[best_r2_idx]:.4f}\n\n"
    summary_text += "📊 TRENDS:\n"
    
    corr_diff = test_corrs[-1] - test_corrs[0]
    if abs(corr_diff) < 0.02:
        summary_text += f"  Corr: Stable\n"
    elif corr_diff > 0:
        summary_text += f"  Corr: ↑ {corr_diff:.3f}\n"
    else:
        summary_text += f"  Corr: ↓ {abs(corr_diff):.3f}\n"
    
    mse_diff = test_mses[-1] - test_mses[0]
    if abs(mse_diff) < 0.001:
        summary_text += f"  MSE: Stable\n"
    elif mse_diff < 0:
        summary_text += f"  MSE: ↓ {abs(mse_diff):.4f}\n"
    else:
        summary_text += f"  MSE: ↑ {mse_diff:.4f}\n"
    
    usage_diff = codebook_usage[-1] - codebook_usage[0]
    summary_text += f"  Usage: {usage_diff:+.1f}%\n"
    
    axes[2, 2].text(0.1, 0.9, summary_text, 
                    transform=axes[2, 2].transAxes,
                    fontsize=11, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Log su W&B
    wandb.log({"analysis/main_plot": wandb.Image(fig)})
    plt.close()
    
    print(f"✅ Main plot loggato su W&B")
    
    # ========================================================================
    # FIGURA 2: Training curves
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"📈 Training Curves - Single Neuron", 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, (result, color) in enumerate(zip(results, colors)):
        size = result['num_embeddings']
        history = result['training_history']
        
        axes[0, 0].plot(history['train_recon_loss'], color=color, 
                       alpha=0.7, linewidth=1.5, label=f'{size}')
        axes[0, 1].plot(history['train_perplexity'], color=color, 
                       alpha=0.7, linewidth=1.5, label=f'{size}')
        
        test_epochs = np.arange(0, len(history['train_loss']), 5)[:len(history['test_mse'])]
        axes[1, 0].plot(test_epochs, history['test_mse'], color=color, 
                       marker='o', alpha=0.7, linewidth=1.5, label=f'{size}')
        axes[1, 1].plot(test_epochs, history['test_correlation'], color=color, 
                       marker='o', alpha=0.7, linewidth=1.5, label=f'{size}')
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Train Recon Loss')
    axes[0, 0].set_title('Training Loss Evolution')
    axes[0, 0].legend(title='Codes', ncol=2, fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].set_title('Perplexity Evolution')
    axes[0, 1].legend(title='Codes', ncol=2, fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Test MSE')
    axes[1, 0].set_title('Test MSE Evolution')
    axes[1, 0].legend(title='Codes', ncol=2, fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Test Correlation')
    axes[1, 1].set_title('Test Correlation Evolution')
    axes[1, 1].legend(title='Codes', ncol=2, fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Log su W&B
    wandb.log({"analysis/training_curves": wandb.Image(fig)})
    plt.close()
    
    print(f"✅ Training curves loggate su W&B")
    """Crea grafici completi dell'analisi"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Estrai metriche
    codebook_sizes = [r['num_embeddings'] for r in results]
    train_losses = [r['final_train_loss'] for r in results]
    test_mses = [r['final_test_metrics']['mse'] for r in results]
    test_corrs = [r['final_test_metrics']['correlation_mean'] for r in results]
    test_r2 = [r['final_test_metrics']['r_squared'] for r in results]
    perplexities = [r['final_test_metrics']['perplexity'] for r in results]
    codebook_usage = [r['codebook_usage_pct'] for r in results]
    num_params = [r['num_parameters'] for r in results]
    
    # ========================================================================
    # FIGURA PRINCIPALE: 3x3 grid
    # ========================================================================
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    title = f"📊 Codebook Size Impact Analysis - {'30 Neurons' if mode == 'multi' else '1 Neuron'}"
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # === ROW 1: Metriche vs Codebook Size ===
    
    # 1. Train Loss
    axes[0, 0].plot(codebook_sizes, train_losses, 'b-o', linewidth=2, markersize=8)
    axes[0, 0].set_xscale('log', base=2)
    axes[0, 0].set_xlabel('Codebook Size', fontsize=11)
    axes[0, 0].set_ylabel('Train Loss (MSE)', fontsize=11)
    axes[0, 0].set_title('Training Loss vs Codebook Size', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(codebook_sizes)
    axes[0, 0].set_xticklabels([str(s) for s in codebook_sizes], rotation=45)
    
    # Annotazione miglior risultato
    best_train_idx = np.argmin(train_losses)
    axes[0, 0].scatter(codebook_sizes[best_train_idx], train_losses[best_train_idx], 
                       color='red', s=200, marker='*', zorder=5, label='Best')
    axes[0, 0].legend()
    
    # 2. Test MSE
    axes[0, 1].plot(codebook_sizes, test_mses, 'r-o', linewidth=2, markersize=8)
    axes[0, 1].set_xscale('log', base=2)
    axes[0, 1].set_xlabel('Codebook Size', fontsize=11)
    axes[0, 1].set_ylabel('Test MSE', fontsize=11)
    axes[0, 1].set_title('Test MSE vs Codebook Size', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(codebook_sizes)
    axes[0, 1].set_xticklabels([str(s) for s in codebook_sizes], rotation=45)
    
    best_mse_idx = np.argmin(test_mses)
    axes[0, 1].scatter(codebook_sizes[best_mse_idx], test_mses[best_mse_idx], 
                       color='darkred', s=200, marker='*', zorder=5, label='Best')
    axes[0, 1].legend()
    
    # 3. Test Correlation
    axes[0, 2].plot(codebook_sizes, test_corrs, 'g-o', linewidth=2, markersize=8)
    axes[0, 2].set_xscale('log', base=2)
    axes[0, 2].set_xlabel('Codebook Size', fontsize=11)
    axes[0, 2].set_ylabel('Correlation', fontsize=11)
    axes[0, 2].set_title('Test Correlation vs Codebook Size', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xticks(codebook_sizes)
    axes[0, 2].set_xticklabels([str(s) for s in codebook_sizes], rotation=45)
    axes[0, 2].set_ylim([0.5, 1.0])
    
    best_corr_idx = np.argmax(test_corrs)
    axes[0, 2].scatter(codebook_sizes[best_corr_idx], test_corrs[best_corr_idx], 
                       color='darkgreen', s=200, marker='*', zorder=5, label='Best')
    axes[0, 2].legend()
    
    # === ROW 2: R², Perplexity, Codebook Usage ===
    
    # 4. R² Score
    axes[1, 0].plot(codebook_sizes, test_r2, 'purple', marker='o', linewidth=2, markersize=8)
    axes[1, 0].set_xscale('log', base=2)
    axes[1, 0].set_xlabel('Codebook Size', fontsize=11)
    axes[1, 0].set_ylabel('R² Score', fontsize=11)
    axes[1, 0].set_title('R² Score vs Codebook Size', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(codebook_sizes)
    axes[1, 0].set_xticklabels([str(s) for s in codebook_sizes], rotation=45)
    axes[1, 0].set_ylim([0, 1.0])
    
    best_r2_idx = np.argmax(test_r2)
    axes[1, 0].scatter(codebook_sizes[best_r2_idx], test_r2[best_r2_idx], 
                       color='darkviolet', s=200, marker='*', zorder=5, label='Best')
    axes[1, 0].legend()
    
    # 5. Perplexity
    axes[1, 1].plot(codebook_sizes, perplexities, 'orange', marker='o', linewidth=2, markersize=8)
    axes[1, 1].set_xscale('log', base=2)
    axes[1, 1].set_xlabel('Codebook Size', fontsize=11)
    axes[1, 1].set_ylabel('Perplexity', fontsize=11)
    axes[1, 1].set_title('Perplexity vs Codebook Size', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(codebook_sizes)
    axes[1, 1].set_xticklabels([str(s) for s in codebook_sizes], rotation=45)
    
    # Linea ideale (perplexity = num_embeddings)
    axes[1, 1].plot(codebook_sizes, codebook_sizes, 'k--', alpha=0.3, label='Ideal')
    axes[1, 1].legend()
    
    # 6. Codebook Usage
    axes[1, 2].plot(codebook_sizes, codebook_usage, 'cyan', marker='o', linewidth=2, markersize=8)
    axes[1, 2].set_xscale('log', base=2)
    axes[1, 2].set_xlabel('Codebook Size', fontsize=11)
    axes[1, 2].set_ylabel('Usage (%)', fontsize=11)
    axes[1, 2].set_title('Codebook Usage vs Size', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xticks(codebook_sizes)
    axes[1, 2].set_xticklabels([str(s) for s in codebook_sizes], rotation=45)
    axes[1, 2].set_ylim([0, 105])
    axes[1, 2].axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100%')
    axes[1, 2].legend()
    
    # === ROW 3: Trade-offs ===
    
    # 7. Accuracy vs Complexity (Params)
    axes[2, 0].scatter(num_params, test_corrs, c=codebook_sizes, 
                       cmap='viridis', s=200, alpha=0.7, edgecolors='black')
    axes[2, 0].set_xlabel('Number of Parameters', fontsize=11)
    axes[2, 0].set_ylabel('Test Correlation', fontsize=11)
    axes[2, 0].set_title('Accuracy vs Model Complexity', fontsize=12, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Annotazioni
    for i, size in enumerate(codebook_sizes):
        axes[2, 0].annotate(str(size), (num_params[i], test_corrs[i]), 
                           fontsize=8, ha='center', va='bottom')
    
    # 8. Correlation vs MSE
    axes[2, 1].scatter(test_mses, test_corrs, c=codebook_sizes, 
                       cmap='plasma', s=200, alpha=0.7, edgecolors='black')
    axes[2, 1].set_xlabel('Test MSE', fontsize=11)
    axes[2, 1].set_ylabel('Test Correlation', fontsize=11)
    axes[2, 1].set_title('Correlation vs MSE Trade-off', fontsize=12, fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    for i, size in enumerate(codebook_sizes):
        axes[2, 1].annotate(str(size), (test_mses[i], test_corrs[i]), 
                           fontsize=8, ha='center', va='bottom')
    
    # 9. Summary Table
    axes[2, 2].axis('off')
    
    # Trova best configurations
    best_corr_size = codebook_sizes[best_corr_idx]
    best_mse_size = codebook_sizes[best_mse_idx]
    best_r2_size = codebook_sizes[best_r2_idx]
    
    summary_text = "🏆 BEST CONFIGURATIONS\n\n"
    summary_text += f"Best Correlation:\n"
    summary_text += f"  {best_corr_size} codes\n"
    summary_text += f"  Corr: {test_corrs[best_corr_idx]:.4f}\n\n"
    
    summary_text += f"Best MSE:\n"
    summary_text += f"  {best_mse_size} codes\n"
    summary_text += f"  MSE: {test_mses[best_mse_idx]:.6f}\n\n"
    
    summary_text += f"Best R²:\n"
    summary_text += f"  {best_r2_size} codes\n"
    summary_text += f"  R²: {test_r2[best_r2_idx]:.4f}\n\n"
    
    summary_text += "📊 TRENDS:\n"
    
    # Trend correlation
    corr_diff = test_corrs[-1] - test_corrs[0]
    if abs(corr_diff) < 0.02:
        summary_text += f"  Corr: Stable\n"
    elif corr_diff > 0:
        summary_text += f"  Corr: ↑ {corr_diff:.3f}\n"
    else:
        summary_text += f"  Corr: ↓ {abs(corr_diff):.3f}\n"
    
    # Trend MSE
    mse_diff = test_mses[-1] - test_mses[0]
    if abs(mse_diff) < 0.001:
        summary_text += f"  MSE: Stable\n"
    elif mse_diff < 0:
        summary_text += f"  MSE: ↓ {abs(mse_diff):.4f}\n"
    else:
        summary_text += f"  MSE: ↑ {mse_diff:.4f}\n"
    
    # Usage trend
    usage_diff = codebook_usage[-1] - codebook_usage[0]
    summary_text += f"  Usage: {usage_diff:+.1f}%\n"
    
    axes[2, 2].text(0.1, 0.9, summary_text, 
                    transform=axes[2, 2].transAxes,
                    fontsize=11, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Salva
    filename = f"codebook_analysis_{mode}_neurons.png"
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot salvato: {save_dir / filename}")
    
    # ========================================================================
    # FIGURA 2: Training curves per ogni codebook size
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"📈 Training Curves - {'30 Neurons' if mode == 'multi' else '1 Neuron'}", 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, (result, color) in enumerate(zip(results, colors)):
        size = result['num_embeddings']
        history = result['training_history']
        
        # Plot train loss
        axes[0, 0].plot(history['train_recon_loss'], color=color, 
                       alpha=0.7, linewidth=1.5, label=f'{size}')
        
        # Plot perplexity
        axes[0, 1].plot(history['train_perplexity'], color=color, 
                       alpha=0.7, linewidth=1.5, label=f'{size}')
        
        # Plot test MSE
        test_epochs = np.arange(0, len(history['train_loss']), 5)[:len(history['test_mse'])]
        axes[1, 0].plot(test_epochs, history['test_mse'], color=color, 
                       marker='o', alpha=0.7, linewidth=1.5, label=f'{size}')
        
        # Plot test correlation
        axes[1, 1].plot(test_epochs, history['test_correlation'], color=color, 
                       marker='o', alpha=0.7, linewidth=1.5, label=f'{size}')
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Train Recon Loss')
    axes[0, 0].set_title('Training Loss Evolution')
    axes[0, 0].legend(title='Codes', ncol=2, fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].set_title('Perplexity Evolution')
    axes[0, 1].legend(title='Codes', ncol=2, fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Test MSE')
    axes[1, 0].set_title('Test MSE Evolution')
    axes[1, 0].legend(title='Codes', ncol=2, fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Test Correlation')
    axes[1, 1].set_title('Test Correlation Evolution')
    axes[1, 1].legend(title='Codes', ncol=2, fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"training_curves_{mode}_neurons.png"
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves salvate: {save_dir / filename}")


def create_comparison_table(results):
    """Crea tabella comparativa dei risultati"""
    
    print(f"\n{'='*100}")
    print(f"📊 RISULTATI COMPLETI - Single Neuron")
    print(f"{'='*100}")
    
    header = (f"{'Codes':<8} | {'Params':<10} | {'Train Loss':<12} | "
              f"{'Test MSE':<12} | {'Corr':<8} | {'R²':<8} | "
              f"{'Perp':<8} | {'Usage':<8}")
    print(header)
    print("-"*100)
    
    for r in results:
        row = (f"{r['num_embeddings']:<8} | "
               f"{r['num_parameters']:<10,} | "
               f"{r['final_train_loss']:<12.6f} | "
               f"{r['final_test_metrics']['mse']:<12.6f} | "
               f"{r['final_test_metrics']['correlation_mean']:<8.4f} | "
               f"{r['final_test_metrics']['r_squared']:<8.4f} | "
               f"{r['final_test_metrics']['perplexity']:<8.1f} | "
               f"{r['codebook_usage_pct']:<8.1f}%")
        print(row)
    
    print("="*100)
    
    # Best configurations
    corrs = [r['final_test_metrics']['correlation_mean'] for r in results]
    mses = [r['final_test_metrics']['mse'] for r in results]
    r2s = [r['final_test_metrics']['r_squared'] for r in results]
    
    best_corr_idx = np.argmax(corrs)
    best_mse_idx = np.argmin(mses)
    best_r2_idx = np.argmax(r2s)
    
    print(f"\n🏆 BEST CONFIGURATIONS:")
    print(f"   Best Correlation: {results[best_corr_idx]['num_embeddings']} codes "
          f"(corr={corrs[best_corr_idx]:.4f})")
    print(f"   Best MSE:         {results[best_mse_idx]['num_embeddings']} codes "
          f"(mse={mses[best_mse_idx]:.6f})")
    print(f"   Best R²:          {results[best_r2_idx]['num_embeddings']} codes "
          f"(R²={r2s[best_r2_idx]:.4f})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analisi impatto codebook size - SINGOLO NEURONE')
    parser.add_argument('--sizes', type=int, nargs='+', default=None,
                       help='Codebook sizes da testare (default: 8 16 32 64 128 256 512 1024 2048)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    if args.sizes:
        codebook_sizes = sorted(args.sizes)
    else:
        codebook_sizes = CODEBOOK_SIZES
    
    print(f"\n{'='*70}")
    print(f"🔬 CODEBOOK SIZE IMPACT ANALYSIS - SINGLE NEURON")
    print(f"{'='*70}")
    print(f"\n📋 Testing {len(codebook_sizes)} configurations:")
    print(f"   Codebook sizes: {codebook_sizes}")
    print(f"   Max epochs: {TRAINING_CONFIG['max_epochs']}")
    print(f"   Training sessions: {len(TRAINING_SESSION_IDS)}")
    
    # W&B
    wandb.init(
        project="calcium-vqvae-codebook-analysis-single-neuron",
        name=f"codebook-sweep-single-{datetime.now().strftime('%m%d-%H%M')}",
        config={
            'mode': 'single',
            'codebook_sizes': codebook_sizes,
            'num_training_sessions': len(TRAINING_SESSION_IDS),
            **TRAINING_CONFIG,
            **MODEL_CONFIG
        },
        tags=['codebook-analysis', 'sweep', 'single-neuron']
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    # Run experiments
    results = []
    
    for i, num_embeddings in enumerate(codebook_sizes, 1):
        print(f"\n{'='*70}")
        print(f"🧪 Experiment {i}/{len(codebook_sizes)}: {num_embeddings} codes")
        print(f"{'='*70}")
        
        try:
            result = train_and_evaluate_codebook_size(
                num_embeddings, 
                device=device
            )
            
            results.append(result)
            
            # Log to W&B
            wandb.log({
                f'codebook_{num_embeddings}/train_loss': result['final_train_loss'],
                f'codebook_{num_embeddings}/test_mse': result['final_test_metrics']['mse'],
                f'codebook_{num_embeddings}/test_correlation': result['final_test_metrics']['correlation_mean'],
                f'codebook_{num_embeddings}/test_r_squared': result['final_test_metrics']['r_squared'],
                f'codebook_{num_embeddings}/perplexity': result['final_test_metrics']['perplexity'],
                f'codebook_{num_embeddings}/codebook_usage': result['codebook_usage_pct'],
                f'codebook_{num_embeddings}/num_parameters': result['num_parameters'],
            })
            
            print(f"✅ Experiment {i} completato!")
            
        except Exception as e:
            print(f"❌ Errore nell'esperimento {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # Analysis
    if len(results) > 0:
        print(f"\n{'='*70}")
        print(f"📊 ANALISI RISULTATI")
        print(f"{'='*70}")
        
        # Tabella
        create_comparison_table(results)
        
        # Plots (log solo su W&B)
        print(f"\n🎨 Creando visualizzazioni e loggando su W&B...")
        plot_codebook_analysis(results)
        
        # Summary su W&B
        best_corr_idx = np.argmax([r['final_test_metrics']['correlation_mean'] for r in results])
        best_mse_idx = np.argmin([r['final_test_metrics']['mse'] for r in results])
        
        wandb.log({
            'summary/best_correlation': results[best_corr_idx]['final_test_metrics']['correlation_mean'],
            'summary/best_correlation_codebook_size': results[best_corr_idx]['num_embeddings'],
            'summary/best_mse': results[best_mse_idx]['final_test_metrics']['mse'],
            'summary/best_mse_codebook_size': results[best_mse_idx]['num_embeddings'],
        })
        
        # Log tabella risultati su W&B
        table_data = []
        for r in results:
            table_data.append([
                r['num_embeddings'],
                r['num_parameters'],
                r['final_train_loss'],
                r['final_test_metrics']['mse'],
                r['final_test_metrics']['correlation_mean'],
                r['final_test_metrics']['r_squared'],
                r['final_test_metrics']['perplexity'],
                r['codebook_usage_pct']
            ])
        
        results_table = wandb.Table(
            columns=["Codes", "Params", "Train Loss", "Test MSE", 
                    "Correlation", "R²", "Perplexity", "Usage %"],
            data=table_data
        )
        
        wandb.log({"results/comparison_table": results_table})
        
        print(f"\n✅ Analisi completata!")
        print(f"   Tutti i risultati su W&B: {wandb.run.url}")
    
    else:
        print(f"\n❌ Nessun esperimento completato con successo")
    
    wandb.finish()
    print("\n✅ W&B run completato!")


if __name__ == "__main__":
    main()