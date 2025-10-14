#!/usr/bin/env python3
"""
FIXED Training script with proper VQ-VAE handling
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import wandb
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from models.vqvae import CalciumVQVAE
from datasets.calcium import (
    create_unified_dataloaders,
    TRAINING_SESSION_IDS,
    TEST_SESSION_IDS
)

# ✅ FIXED MODEL CONFIG - ANTI-COLLAPSE
MODEL_CONFIG = {
    'num_neurons': 30,
    'num_hiddens': 512,
    'num_residual_layers': 6,
    'num_residual_hiddens': 256,
    'num_embeddings': 16,
    'embedding_dim': 32,
    'commitment_cost': 1.0,  # ✅ AUMENTATO per evitare collapse
    'dropout_rate': 0.0,
    'use_quantizer': True,
}

def create_enhanced_calcium_vqvae():
    """Create model with PROPER quantizer settings"""
    return CalciumVQVAE(
        num_neurons=MODEL_CONFIG['num_neurons'],
        num_hiddens=MODEL_CONFIG['num_hiddens'],
        num_residual_layers=MODEL_CONFIG['num_residual_layers'],
        num_residual_hiddens=MODEL_CONFIG['num_residual_hiddens'],
        num_embeddings=MODEL_CONFIG['num_embeddings'],
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        commitment_cost=MODEL_CONFIG['commitment_cost'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        use_quantizer=MODEL_CONFIG['use_quantizer']
    )

def train_epoch_enhanced(model, dataloader, optimizer, device, epoch, config):
    """Training with PROPER VQ SCHEDULING"""
    model.train()
    
    # Disable dropout explicitly
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    
    # ✅ IMPROVED VQ WEIGHT SCHEDULING
    warmup_epochs = config.get('warmup_epochs', 20)  # Più warmup
    max_epochs = config.get('max_epochs', 200)
    vq_weight_start = config.get('vq_weight_start', 0.0)
    vq_weight_final = config.get('vq_weight_final', 0.1)  # Target più alto
    
    if epoch < warmup_epochs:
        # Warmup: peso VQ cresce gradualmente da 0
        progress = epoch / warmup_epochs
        vq_weight = vq_weight_start + progress * (vq_weight_final - vq_weight_start) * 0.1
        phase = "WARMUP"
    else:
        # Post-warmup: aumenta fino a vq_weight_final
        progress = (epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
        vq_weight = vq_weight_start + progress * vq_weight_final
        vq_weight = min(vq_weight, vq_weight_final)
        phase = "VQ_ACTIVE"
    
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    
    for neural_data in dataloader:
        neural_data = neural_data.to(device)
        
        # ✅ Forward returns 5 values now
        vq_loss, recon, perplexity, _, _ = model(neural_data)
        
        # ✅ HUBER LOSS per reconstruction
        diff = torch.abs(recon - neural_data)
        delta = 1.0
        recon_loss = torch.where(
            diff < delta,
            0.5 * diff ** 2,
            delta * (diff - 0.5 * delta)
        ).mean()
        
        # ✅ VARIANCE PRESERVATION
        var_original = torch.var(neural_data)
        var_recon = torch.var(recon)
        var_loss = (var_original - var_recon) ** 2
        
        # ✅ COMBINED LOSS with scheduled VQ
        loss = recon_loss + 0.5 * var_loss + vq_weight * vq_loss
        
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
        'train/total_loss': total_loss / num_batches,
        'train/recon_loss': total_recon_loss / num_batches,
        'train/vq_loss': total_vq_loss / num_batches,
        'train/perplexity': total_perplexity / num_batches,
        'train/vq_weight': vq_weight,
        'train/phase': phase
    }

def evaluate_model_enhanced(model, dataloader, device):
    """Evaluation with proper handling"""
    model.eval()
    all_original = []
    all_reconstructed = []
    total_vq_loss = 0
    total_perplexity = 0
    num_batches = 0
    
    with torch.no_grad():
        for neural_data in dataloader:
            neural_data = neural_data.to(device)
            
            # ✅ Unpack 5 values
            vq_loss, recon, perplexity, _, _ = model(neural_data)
            
            all_original.append(neural_data.cpu().numpy())
            all_reconstructed.append(recon.cpu().numpy())
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
    
    original = np.concatenate(all_original, axis=0)
    reconstructed = np.concatenate(all_reconstructed, axis=0)
    
    # MSE
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    
    # R²
    y_true = original.flatten()
    y_pred = reconstructed.flatten()
    y_mean = np.mean(y_true)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    
    # Per-neuron correlations
    correlations = []
    for sample_idx in range(original.shape[0]):
        for neuron_idx in range(original.shape[1]):
            orig_neuron = original[sample_idx, neuron_idx, :]
            recon_neuron = reconstructed[sample_idx, neuron_idx, :]
            
            if len(orig_neuron) > 0 and np.std(orig_neuron) > 1e-8 and np.std(recon_neuron) > 1e-8:
                corr, _ = pearsonr(orig_neuron, recon_neuron)
                correlations.append(corr if not np.isnan(corr) else 0)
    
    perfect_neurons = np.sum(np.array(correlations) > 0.99)
    excellent_neurons = np.sum(np.array(correlations) > 0.95)
    good_neurons = np.sum(np.array(correlations) > 0.5)
    
    return {
        'test/mse': mse,
        'test/mae': mae,
        'test/r_squared': r_squared,
        'test/pearson_mean': np.mean(correlations) if correlations else 0,
        'test/pearson_min': np.min(correlations) if correlations else 0,
        'test/pearson_max': np.max(correlations) if correlations else 1,
        'test/perfect_neurons': perfect_neurons,
        'test/perfect_neurons_pct': (perfect_neurons / len(correlations) * 100) if correlations else 0,
        'test/excellent_neurons': excellent_neurons,
        'test/excellent_neurons_pct': (excellent_neurons / len(correlations) * 100) if correlations else 0,
        'test/good_neurons': good_neurons,
        'test/good_neurons_pct': (good_neurons / len(correlations) * 100) if correlations else 0,
        'test/vq_loss': total_vq_loss / num_batches if num_batches > 0 else 0,
        'test/perplexity': total_perplexity / num_batches if num_batches > 0 else 0,
    }

def log_reconstructions_enhanced(original, reconstructed, epoch, num_examples=2):
    """Log reconstructions to W&B"""
    for i in range(min(num_examples, original.shape[0])):
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        
        neuron_vars = np.var(original[i], axis=1)
        top_neurons = np.argsort(neuron_vars)[-15:]
        
        im1 = axes[0,0].imshow(original[i, top_neurons, :], aspect='auto', cmap='viridis')
        axes[0,0].set_title('Original')
        axes[0,0].set_ylabel('Neurons')
        
        im2 = axes[0,1].imshow(reconstructed[i, top_neurons, :], aspect='auto', cmap='viridis')
        axes[0,1].set_title('Reconstruction')
        
        error = np.abs(original[i, top_neurons, :] - reconstructed[i, top_neurons, :])
        im3 = axes[1,0].imshow(error, aspect='auto', cmap='Reds')
        axes[1,0].set_title('Absolute Error')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Neurons')
        
        correlations = []
        for neuron_idx in top_neurons:
            corr, _ = pearsonr(original[i, neuron_idx, :], reconstructed[i, neuron_idx, :])
            correlations.append(corr if not np.isnan(corr) else 0)
        
        axes[1,1].bar(range(len(correlations)), correlations)
        axes[1,1].set_title('Per-Neuron Correlation')
        axes[1,1].set_xlabel('Neuron Index')
        axes[1,1].set_ylabel('Correlation')
        axes[1,1].axhline(y=0.99, color='r', linestyle='--', label='Perfect (0.99)')
        axes[1,1].axhline(y=0.95, color='orange', linestyle='--', label='Excellent (0.95)')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        try:
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            pil_image = Image.open(buf)
            wandb.log({f"reconstructions/epoch_{epoch}_example_{i}": wandb.Image(pil_image)})
            buf.close()
        except Exception as e:
            print(f"⚠️ W&B logging failed: {e}")
        finally:
            plt.close(fig)

def main():
    wandb.init(
        project="calcium-vqvae-multi-session",
        name="FIXED-VQ-proper-scheduling",
        config={
            "model_type": "CalciumVQVAE_FixedVQ",
            "num_sessions_train": len(TRAINING_SESSION_IDS),
            "num_sessions_test": len(TEST_SESSION_IDS),
            "window_size": 60,
            "stride": 50,
            
            # Training parameters
            "batch_size": 32,  # ✅ Maggiore per stabilità
            "learning_rate": 0.0003,
            "max_epochs": 105,
            "patience": 50,
            
            # VQ SCHEDULING ✅ IMPROVED
            "warmup_epochs": 20,
            "vq_weight_start": 0.0,
            "vq_weight_final": 0.3,  # ← AUMENTA (era 0.1)
            
            **MODEL_CONFIG
        }
    )
    
    print("🎯 FIXED VQ-VAE TRAINING")
    print("="*70)
    print(f"✅ Commitment cost: {MODEL_CONFIG['commitment_cost']}")
    print(f"✅ VQ weight final: {wandb.config.vq_weight_final}")
    print(f"✅ Warmup epochs: {wandb.config.warmup_epochs}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"💻 Using device: {device}")
    
    try:
        # Dataset loading
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
        print(f"   Sessions: {dataset_info['num_sessions_train']}")
        print(f"   Train: {dataset_info['train_samples']}")
        print(f"   Test: {dataset_info['test_samples']}")
        
        # Create model
        print("\n🏗️ Creating FIXED model...")
        model = create_enhanced_calcium_vqvae().to(device)
        
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
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )
        
        # Training loop
        print("\n🚀 Starting training...")
        best_test_mse = float('inf')
        best_train_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(wandb.config.max_epochs):
            train_metrics = train_epoch_enhanced(
                model, train_loader, optimizer, device, 
                epoch, wandb.config
            )
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            if epoch % 5 == 0:
                test_metrics = evaluate_model_enhanced(model, test_loader, device)
                current_test_mse = test_metrics['test/mse']
                current_train_loss = train_metrics['train/recon_loss']
                
                all_metrics = {
                    **train_metrics, 
                    **test_metrics, 
                    'epoch': epoch,
                    'learning_rate': current_lr
                }
                wandb.log(all_metrics)
                
                # ✅ Log codebook usage
                if hasattr(model, 'get_codebook_usage'):
                    usage_stats = model.get_codebook_usage()
                    wandb.log({
                        'codebook/usage_pct': usage_stats.get('usage_percentage', 0),
                        'codebook/used_codes': usage_stats.get('used_codes', 0)
                    })
                
                print(f"Epoch {epoch:3d} [{train_metrics['train/phase']}]:")
                print(f"  Train Loss={current_train_loss:.6f}, VQ={train_metrics['train/vq_loss']:.6f}")
                print(f"  Test MSE={current_test_mse:.6f}, R²={test_metrics['test/r_squared']:.4f}")
                print(f"  Perplexity={train_metrics['train/perplexity']:.1f}")
                print(f"  Perfect={test_metrics['test/perfect_neurons_pct']:.1f}%")
                
                if epoch % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        batch_data = next(iter(test_loader))
                        sample_batch = batch_data.to(device) if not isinstance(batch_data, (list, tuple)) else batch_data[0].to(device)
                        
                        _, recon_sample, _, _, _ = model(sample_batch)
                        
                        log_reconstructions_enhanced(
                            sample_batch.cpu().numpy(), 
                            recon_sample.cpu().numpy(),
                            epoch
                        )
                
                if current_train_loss < best_train_loss:
                    best_train_loss = current_train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= wandb.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                wandb.log({**train_metrics, 'epoch': epoch, 'learning_rate': current_lr})
        
        # Final evaluation
        print("\n🎯 FINAL EVALUATION:")
        final_metrics = evaluate_model_enhanced(model, test_loader, device)
        final_train_metrics = train_epoch_enhanced(
            model, train_loader, optimizer, device,
            epoch if 'epoch' in locals() else wandb.config.max_epochs - 1,
            wandb.config
        )
        
        print(f"Train Loss: {final_train_metrics['train/recon_loss']:.6f}")
        print(f"Test MSE: {final_metrics['test/mse']:.6f}")
        print(f"R²: {final_metrics['test/r_squared']:.4f}")
        print(f"Perfect neurons: {final_metrics['test/perfect_neurons_pct']:.1f}%")
        
        wandb.log({
            'final/train_recon_loss': final_train_metrics['train/recon_loss'],
            'final/test_mse': final_metrics['test/mse'],
            'final/r_squared': final_metrics['test/r_squared'],
            'final/perfect_neurons_pct': final_metrics['test/perfect_neurons_pct'],
        })
        
        # Save model
        print("\n💾 Saving model...")
        os.makedirs('./results', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch if 'epoch' in locals() else wandb.config.max_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': MODEL_CONFIG,
            'dataset_info': dataset_info,
            'training_sessions': TRAINING_SESSION_IDS,
            'test_sessions': TEST_SESSION_IDS,
            'final_metrics': final_metrics
        }
        
        torch.save(checkpoint, './results/best_multi_session_FIXED_VQ_16.pth')
        print("✅ Model saved!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()