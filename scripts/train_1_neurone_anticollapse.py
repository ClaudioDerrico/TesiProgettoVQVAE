#!/usr/bin/env python3
"""
ANTI-COLLAPSE Training for Single Neuron VQ-VAE

CRITICAL FIXES:
1. Codebook SMALL (16 embeddings) ✓ manteniamo
2. Commitment cost HIGH (1.0 invece di 0.25)
3. VQ weight scheduling AGGRESSIVE
4. Diversity loss to prevent collapse
5. NO dropout during training
6. Codebook re-initialization with Xavier
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import numpy as np
import wandb
import matplotlib.pyplot as plt
from PIL import Image

from models.vqvae import CalciumVQVAE
from datasets.calcium import (
    create_unified_dataloaders,
    TRAINING_SESSION_IDS,
    TEST_SESSION_IDS
)

# ✅ CONFIGURAZIONE ANTI-COLLAPSE
MODEL_CONFIG = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 16,         # ← Manteniamo 16 (già abbastanza piccolo)
    'embedding_dim': 32,
    'commitment_cost': 1.0,       # ← AUMENTATO da 0.25 a 1.0 (4x)
    'dropout_rate': 0.0,          # ← NO DROPOUT (era 0.1)
    'use_quantizer': True,
}

def initialize_codebook_properly(model):
    """
    Re-inizializza il codebook con Xavier e scala appropriata
    """
    print("🔧 Re-inizializzando codebook con Xavier...")
    
    nn.init.xavier_uniform_(model.vector_quantization.embedding.weight.data)
    model.vector_quantization.embedding.weight.data *= 0.3  # Scale down per evitare valori estremi
    
    mean = model.vector_quantization.embedding.weight.data.mean().item()
    std = model.vector_quantization.embedding.weight.data.std().item()
    
    print(f"✅ Codebook re-inizializzato:")
    print(f"   Mean: {mean:.4f}")
    print(f"   Std:  {std:.4f}")


def compute_diversity_loss(model):
    """
    Loss che PENALIZZA codici troppo simili nel codebook.
    Forza il modello a usare codici diversi.
    """
    embeddings = model.vector_quantization.embedding.weight
    
    # Normalizza embeddings
    embed_norms = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
    
    # Calcola similarity matrix (cosine similarity tra tutti i codici)
    similarity_matrix = torch.matmul(embed_norms, embed_norms.t())
    
    # Rimuovi diagonale (self-similarity = 1.0)
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
    similarity_matrix = similarity_matrix * (1 - mask)
    
    # Penalizza alta similarità tra codici diversi
    diversity_loss = similarity_matrix.abs().mean()
    
    return diversity_loss


def train_epoch_anti_collapse(model, dataloader, optimizer, device, epoch, config):
    """
    Training con ANTI-COLLAPSE measures
    """
    model.train()
    
    # ✅ VQ WEIGHT SCHEDULING AGGRESSIVO
    warmup_epochs = config.get('warmup_epochs', 5)  # ← BREVE warmup
    max_epochs = config.get('max_epochs', 200)
    
    max_vq = config.get('vq_weight_max', 0.1)
    
    if epoch < warmup_epochs:
        # Warmup: VQ weight cresce linearmente da 0
        vq_weight = (epoch / warmup_epochs) * max_vq * 0.1  # Inizio molto soft
        phase = "WARMUP (VQ OFF)"
    else:
        # Post-warmup: cresce velocemente
        progress = (epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
        # Ramp VELOCE (3x più veloce del normale)
        vq_weight = max_vq * min(1.0, progress * 3.0)
        phase = f"VQ_ACTIVE ({vq_weight:.4f})"
    
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_diversity_loss = 0
    total_perplexity = 0
    
    for neural_data in dataloader:
        neural_data = neural_data.to(device)
        
        # Forward pass
        vq_loss, recon, perplexity, _, _ = model(neural_data)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, neural_data)
        
        # ✅ DIVERSITY LOSS (anti-collapse)
        diversity_loss = compute_diversity_loss(model)
        
        # Combined loss
        loss = recon_loss + vq_weight * vq_loss + 0.05 * diversity_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_diversity_loss += diversity_loss.item()
        total_perplexity += perplexity.item()
    
    num_batches = len(dataloader)
    
    return {
        'train/total_loss': total_loss / num_batches,
        'train/recon_loss': total_recon_loss / num_batches,
        'train/vq_loss': total_vq_loss / num_batches,
        'train/diversity_loss': total_diversity_loss / num_batches,
        'train/perplexity': total_perplexity / num_batches,
        'train/vq_weight': vq_weight,
        'train/phase': phase
    }


def evaluate_model_with_codebook_usage(model, dataloader, device):
    """
    Evaluation + codebook usage tracking
    """
    model.eval()
    all_original = []
    all_reconstructed = []
    all_encoding_indices = []
    total_vq_loss = 0
    total_perplexity = 0
    num_batches = 0
    
    with torch.no_grad():
        for neural_data in dataloader:
            neural_data = neural_data.to(device)
            
            vq_loss, recon, perplexity, _, encoding_indices = model(neural_data)
            
            all_original.append(neural_data.cpu().numpy())
            all_reconstructed.append(recon.cpu().numpy())
            all_encoding_indices.extend(encoding_indices.cpu().numpy().flatten().tolist())
            
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
    
    original = np.concatenate(all_original, axis=0)
    reconstructed = np.concatenate(all_reconstructed, axis=0)
    
    # Metriche ricostruzione
    mse = np.mean((original - reconstructed) ** 2)
    
    # Correlation
    correlations = []
    for sample_idx in range(original.shape[0]):
        orig_trace = original[sample_idx, 0, :]
        recon_trace = reconstructed[sample_idx, 0, :]
        
        if np.std(orig_trace) > 1e-8 and np.std(recon_trace) > 1e-8:
            corr, _ = pearsonr(orig_trace, recon_trace)
            correlations.append(corr if not np.isnan(corr) else 0)
    
    # ✅ CODEBOOK USAGE
    from collections import Counter
    code_counts = Counter(all_encoding_indices)
    num_codes_used = len(code_counts)
    usage_percentage = (num_codes_used / MODEL_CONFIG['num_embeddings']) * 100
    
    # ✅ Top 3 concentration check
    total_assignments = len(all_encoding_indices)
    top_3_counts = sum([count for _, count in code_counts.most_common(3)])
    top_3_concentration = (top_3_counts / total_assignments) * 100
    
    return {
        'test/mse': mse,
        'test/correlation_mean': np.mean(correlations) if correlations else 0,
        'test/correlation_std': np.std(correlations) if correlations else 0,
        'test/vq_loss': total_vq_loss / num_batches if num_batches > 0 else 0,
        'test/perplexity': total_perplexity / num_batches if num_batches > 0 else 0,
        'test/codebook_usage': usage_percentage,
        'test/codes_used': num_codes_used,
        'test/top3_concentration': top_3_concentration,
    }


def visualize_single_neuron_reconstruction(original, reconstructed, epoch):
    """Visualizzazione tracce singolo neurone"""
    
    num_samples = original.shape[0]
    
    # Seleziona sample con ampiezza massima
    amplitudes = []
    for i in range(num_samples):
        trace = original[i, 0, :]
        amplitude = np.max(trace) - np.min(trace)
        amplitudes.append(amplitude)
    
    amplitudes = np.array(amplitudes)
    top_indices = np.argsort(amplitudes)[-4:][::-1]
    
    num_examples = len(top_indices)
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(18, 3*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for plot_idx, sample_idx in enumerate(top_indices):
        orig_trace = original[sample_idx, 0, :]
        recon_trace = reconstructed[sample_idx, 0, :]
        
        time_points = np.arange(len(orig_trace))
        amplitude = amplitudes[sample_idx]
        
        # Plot 1: Overlay
        axes[plot_idx, 0].plot(time_points, orig_trace, 'b-', 
                               label='Original', alpha=0.7, linewidth=1.5)
        axes[plot_idx, 0].plot(time_points, recon_trace, 'r-', 
                               label='Reconstruction', alpha=0.7, linewidth=1.5)
        axes[plot_idx, 0].set_xlabel('Time')
        axes[plot_idx, 0].set_ylabel('Activity')
        axes[plot_idx, 0].set_title(f'Sample {sample_idx}: Amplitude={amplitude:.2f}')
        axes[plot_idx, 0].legend(loc='upper right', fontsize=8)
        axes[plot_idx, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error
        error = np.abs(orig_trace - recon_trace)
        axes[plot_idx, 1].plot(time_points, error, 'k-', alpha=0.7, linewidth=1)
        axes[plot_idx, 1].fill_between(time_points, 0, error, alpha=0.3, color='red')
        axes[plot_idx, 1].set_xlabel('Time')
        axes[plot_idx, 1].set_ylabel('Absolute Error')
        axes[plot_idx, 1].set_title(f'Sample {sample_idx}: Error')
        axes[plot_idx, 1].grid(True, alpha=0.3)
        
        # Plot 3: Correlation scatter
        if np.std(orig_trace) > 1e-8 and np.std(recon_trace) > 1e-8:
            corr, _ = pearsonr(orig_trace, recon_trace)
        else:
            corr = 0
        
        axes[plot_idx, 2].scatter(orig_trace, recon_trace, alpha=0.5, s=20)
        min_val = min(np.min(orig_trace), np.min(recon_trace))
        max_val = max(np.max(orig_trace), np.max(recon_trace))
        axes[plot_idx, 2].plot([min_val, max_val], [min_val, max_val], 
                               'r--', alpha=0.5, label='Perfect')
        axes[plot_idx, 2].set_xlabel('Original')
        axes[plot_idx, 2].set_ylabel('Reconstructed')
        axes[plot_idx, 2].set_title(f'Correlation: {corr:.3f}')
        axes[plot_idx, 2].legend(loc='upper left', fontsize=8)
        axes[plot_idx, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Single Neuron Reconstructions - Epoch {epoch}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    pil_image = Image.fromarray(buf)
    
    return pil_image


def main():
    wandb.init(
        project="calcium-vqvae-single-neuron-FIXED",
        name="anti-collapse-16codes",
        config={
            "model_type": "CalciumVQVAE_AntiCollapse",
            "num_sessions_train": len(TRAINING_SESSION_IDS),
            "window_size": 60,
            "stride": 50,
            
            # Training
            "batch_size": 64,
            "learning_rate": 0.0005,
            "max_epochs": 200,
            "patience": 40,
            "warmup_epochs": 5,  # ← BREVE warmup
            
            # Anti-collapse
            "diversity_weight": 0.05,
            "vq_weight_max": 0.1,
            
            **MODEL_CONFIG
        }
    )
    
    print("="*70)
    print("🛡️ ANTI-COLLAPSE TRAINING: Single Neuron VQ-VAE")
    print("="*70)
    print(f"✅ Codebook: {MODEL_CONFIG['num_embeddings']} codes")
    print(f"✅ Commitment cost: {MODEL_CONFIG['commitment_cost']}")
    print(f"✅ VQ weight max: {wandb.config.vq_weight_max}")
    print(f"✅ Diversity loss: ENABLED (weight={wandb.config.diversity_weight})")
    print(f"✅ Dropout: DISABLED")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"💻 Device: {device}")
    
    try:
        # Dataset
        print("\n📊 Loading datasets...")
        train_loader, test_loader, dataset_info = create_unified_dataloaders(
            train_session_ids=TRAINING_SESSION_IDS,
            batch_size=wandb.config.batch_size,
            test_split=0.2,
            window_size=wandb.config.window_size,
            stride=wandb.config.stride,
            min_neurons=wandb.config.num_neurons,
            num_workers=0,
            use_cross_session_test=False
        )
        
        print(f"✅ Dataset loaded:")
        print(f"   Train: {dataset_info['train_samples']}")
        print(f"   Test: {dataset_info['test_samples']}")
        
        # Model
        print("\n🏗️ Creating model...")
        model = CalciumVQVAE(
            num_neurons=MODEL_CONFIG['num_neurons'],
            num_hiddens=MODEL_CONFIG['num_hiddens'],
            num_residual_layers=MODEL_CONFIG['num_residual_layers'],
            num_residual_hiddens=MODEL_CONFIG['num_residual_hiddens'],
            num_embeddings=MODEL_CONFIG['num_embeddings'],
            embedding_dim=MODEL_CONFIG['embedding_dim'],
            commitment_cost=MODEL_CONFIG['commitment_cost'],
            dropout_rate=MODEL_CONFIG['dropout_rate'],
            use_quantizer=MODEL_CONFIG['use_quantizer']
        ).to(device)
        
        # ✅ RE-INITIALIZE CODEBOOK
        initialize_codebook_properly(model)
        
        num_params = sum(p.numel() for p in model.parameters())
        wandb.log({"model/num_parameters": num_params})
        print(f"✅ Model: {num_params:,} parameters")
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=wandb.config.learning_rate,
            weight_decay=0.0001,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=wandb.config.max_epochs,
            eta_min=1e-6
        )
        
        # Training loop
        print("\n🚀 Starting training...")
        best_test_corr = -1
        best_codebook_usage = 0
        patience_counter = 0
        
        for epoch in range(wandb.config.max_epochs):
            # Train
            train_metrics = train_epoch_anti_collapse(
                model, train_loader, optimizer, device, 
                epoch, wandb.config
            )
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Evaluate every 5 epochs
            if epoch % 5 == 0:
                test_metrics = evaluate_model_with_codebook_usage(
                    model, test_loader, device
                )
                
                all_metrics = {
                    **train_metrics, 
                    **test_metrics, 
                    'epoch': epoch,
                    'learning_rate': current_lr
                }
                wandb.log(all_metrics)
                
                print(f"Epoch {epoch:3d} [{train_metrics['train/phase']}]:")
                print(f"  Recon: {train_metrics['train/recon_loss']:.6f}")
                print(f"  Corr:  {test_metrics['test/correlation_mean']:.4f}")
                print(f"  📊 CODEBOOK: {test_metrics['test/codes_used']}/{MODEL_CONFIG['num_embeddings']} used ({test_metrics['test/codebook_usage']:.1f}%)")
                print(f"  📊 Top3 conc: {test_metrics['test/top3_concentration']:.1f}%")
                print(f"  Perp:  {train_metrics['train/perplexity']:.1f}")
                
                # Visualize every 20 epochs
                if epoch % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        sample_batch = next(iter(test_loader)).to(device)
                        _, recon_sample, _, _, _ = model(sample_batch)
                        
                        img = visualize_single_neuron_reconstruction(
                            sample_batch.cpu().numpy(), 
                            recon_sample.cpu().numpy(),
                            epoch
                        )
                        wandb.log({f"reconstructions/epoch_{epoch}": wandb.Image(img)})
                
                # Check improvement (prioritize codebook usage + correlation)
                current_test_corr = test_metrics['test/correlation_mean']
                current_usage = test_metrics['test/codebook_usage']
                
                # Metrica combinata: buona correlazione + buon usage
                combined_metric = current_test_corr * (current_usage / 100.0)
                best_combined = best_test_corr * (best_codebook_usage / 100.0)
                
                if combined_metric > best_combined or (current_usage > best_codebook_usage and current_test_corr > 0.8):
                    best_test_corr = current_test_corr
                    best_codebook_usage = current_usage
                    patience_counter = 0
                    
                    # Save
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_config': MODEL_CONFIG,
                        'best_correlation': best_test_corr,
                        'best_codebook_usage': best_codebook_usage,
                        'test_metrics': test_metrics
                    }
                    
                    os.makedirs('./results', exist_ok=True)
                    torch.save(checkpoint, './results/best_single_neuron_ANTI_COLLAPSE_16.pth')
                    print(f"  💾 Saved (corr={best_test_corr:.4f}, usage={best_codebook_usage:.1f}%)")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= wandb.config.patience:
                    print(f"Early stopping at epoch {epoch}")
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