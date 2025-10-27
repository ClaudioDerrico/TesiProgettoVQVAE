#!/usr/bin/env python3
"""
Training ANTI-COLLAPSE per Single Neuron VQ-VAE

MODIFICHE CRITICHE rispetto al vecchio training:
1. Codebook PICCOLO (16 invece di 512)
2. VQ weight scheduling CORRETTO (max 0.001)
3. Diversity loss per evitare collapse
4. Codebook re-initialization con Xavier
5. NO dropout
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
    'num_embeddings': 16,         # ← RIDOTTO da 512!
    'embedding_dim': 32,          # ← RIDOTTO da 128!
    'commitment_cost': 0.25,      # ← RIDOTTO da 1.0!
    'dropout_rate': 0.0,          # ← NO dropout!
    'use_quantizer': True,
}

def initialize_codebook_properly(model):
    """
    Re-inizializza il codebook con Xavier e scala
    """
    print("🔧 Re-inizializzando codebook...")
    
    nn.init.xavier_uniform_(model.vector_quantization.embedding.weight.data)
    model.vector_quantization.embedding.weight.data *= 0.3  # Scale down
    
    print(f"✅ Codebook re-inizializzato:")
    print(f"   Mean: {model.vector_quantization.embedding.weight.data.mean():.4f}")
    print(f"   Std:  {model.vector_quantization.embedding.weight.data.std():.4f}")


def compute_diversity_loss(model):
    """
    Loss che PENALIZZA codici troppo simili
    """
    embeddings = model.vector_quantization.embedding.weight
    
    # Normalizza embeddings
    embed_norms = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
    
    # Calcola similarity matrix
    similarity_matrix = torch.matmul(embed_norms, embed_norms.t())
    
    # Rimuovi diagonale (self-similarity)
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
    similarity_matrix = similarity_matrix * (1 - mask)
    
    # Penalizza alta similarità
    diversity_loss = similarity_matrix.abs().mean()
    
    return diversity_loss


def train_epoch_anti_collapse(model, dataloader, optimizer, device, epoch, config):
    """
    Training con ANTI-COLLAPSE measures
    """
    model.train()
    
    # ✅ VQ WEIGHT SCHEDULING CORRETTO
    warmup_epochs = config.get('warmup_epochs', 50)
    max_epochs = config.get('max_epochs', 200)
    
    max_vq = config.get('vq_weight_max', 0.1)
    if epoch < warmup_epochs:
        vq_weight = 0.0
        phase = "WARMUP (VQ OFF)"
    else:
        progress = (epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
        # ramp veloce ma controllata (sale 5 volte più in fretta)
        vq_weight = max_vq * min(1.0, progress * 5.0)
        phase = f"VQ_ACTIVE ({vq_weight:.4f})"
    
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_diversity_loss = 0
    total_perplexity = 0
    
    for neural_data in dataloader:
        neural_data = neural_data.to(device)
        
        # Forward
        vq_loss, recon, perplexity, _, _ = model(neural_data)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, neural_data)
        
        # ✅ DIVERSITY LOSS (anti-collapse)
        diversity_loss = compute_diversity_loss(model)
        
        # Combined loss
        loss = recon_loss + vq_weight * vq_loss + 0.1 * diversity_loss
        
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
    
    # Metriche
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
    
    return {
        'test/mse': mse,
        'test/correlation_mean': np.mean(correlations) if correlations else 0,
        'test/correlation_std': np.std(correlations) if correlations else 0,
        'test/vq_loss': total_vq_loss / num_batches if num_batches > 0 else 0,
        'test/perplexity': total_perplexity / num_batches if num_batches > 0 else 0,
        'test/codebook_usage': usage_percentage,
        'test/codes_used': num_codes_used,
    }


def main():
    wandb.init(
        project="calcium-vqvae-anti-collapse",
        name="single-neuron-16codes-FIXED",
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
            "warmup_epochs": 5,  # ← LUNGO warmup
            
            # Anti-collapse
            "diversity_weight": 0.05,
            "vq_weight_max": 0.1,  # ← BASSO!
            
            **MODEL_CONFIG
        }
    )
    
    print("="*70)
    print("🛡️ ANTI-COLLAPSE TRAINING: Single Neuron VQ-VAE")
    print("="*70)
    print(f"✅ Codebook: {MODEL_CONFIG['num_embeddings']} codes")
    print(f"✅ VQ weight max: 0.001")
    print(f"✅ Diversity loss: ENABLED")
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
        patience_counter = 0
        
        for epoch in range(wandb.config.max_epochs):
            # Train
            train_metrics = train_epoch_anti_collapse(
                model, train_loader, optimizer, device, 
                epoch, wandb.config
            )
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Evaluate
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
                print(f"  Perp:  {train_metrics['train/perplexity']:.1f}")
                
                # Check improvement
                current_test_corr = test_metrics['test/correlation_mean']
                if current_test_corr > best_test_corr:
                    best_test_corr = current_test_corr
                    patience_counter = 0
                    
                    # Save
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_config': MODEL_CONFIG,
                        'best_correlation': best_test_corr,
                        'test_metrics': test_metrics
                    }
                    
                    os.makedirs('./results', exist_ok=True)
                    torch.save(checkpoint, './results/best_single_neuron_ANTI_COLLAPSE_16.pth')
                    print(f"  💾 Saved (corr={best_test_corr:.4f}, usage={test_metrics['test/codebook_usage']:.1f}%)")
                else:
                    patience_counter += 1
                
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