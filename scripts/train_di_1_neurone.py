#!/usr/bin/env python3
"""
FIXED Training script for SINGLE NEURON with proper architecture
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

from models.vqvae import CalciumVQVAE
from datasets.calcium import (
    create_unified_dataloaders,
    TRAINING_SESSION_IDS,
    TEST_SESSION_IDS
)

# ✅ ARCHITETTURA OTTIMIZZATA PER 1 NEURONE
MODEL_CONFIG = {
    'num_neurons': 1,           # 1 neurone
    'num_hiddens': 128,         # ← AUMENTATO (era 64)
    'num_residual_layers': 3,   # ← AUMENTATO (era 2)
    'num_residual_hiddens': 64, # ← AUMENTATO (era 32)
    'num_embeddings': 32,       # ← AUMENTATO (era 16)
    'embedding_dim': 32,        # ← AUMENTATO (era 16)
    'commitment_cost': 0.25,    # ← RIDOTTO (era 1.0)
    'dropout_rate': 0.1,        # ← AGGIUNTO dropout leggero
    'use_quantizer': True,
}

def create_single_neuron_vqvae():
    """Create model optimized for single neuron"""
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

def compute_loss_with_temporal_smoothness(recon, target, alpha=0.1):
    """
    Loss function with temporal smoothness regularization
    Important for single neuron time series
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, target)
    
    # Temporal smoothness loss (penalize large temporal differences)
    if recon.shape[2] > 1:  # Check temporal dimension
        diff_recon = recon[:, :, 1:] - recon[:, :, :-1]
        diff_target = target[:, :, 1:] - target[:, :, :-1]
        smooth_loss = F.mse_loss(diff_recon, diff_target)
    else:
        smooth_loss = 0.0
    
    # Combined loss
    total_loss = recon_loss + alpha * smooth_loss
    
    return total_loss, recon_loss, smooth_loss

def train_epoch_single_neuron(model, dataloader, optimizer, device, epoch, config):
    """Training optimized for single neuron"""
    model.train()
    
    # VQ Weight Scheduling - più graduale
    warmup_epochs = config.get('warmup_epochs', 30)
    max_epochs = config.get('max_epochs', 200)
    
    if epoch < warmup_epochs:
        # Warmup phase: focus on reconstruction
        vq_weight = 0.001 * (epoch / warmup_epochs)
        phase = "WARMUP"
    else:
        # Post-warmup: gradually increase VQ
        progress = (epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
        vq_weight = 0.001 + 0.1 * progress  # Max 0.101
        phase = "VQ_ACTIVE"
    
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    total_smooth_loss = 0
    
    for neural_data in dataloader:
        neural_data = neural_data.to(device)
        
        # Forward pass
        vq_loss, recon, perplexity, _, _ = model(neural_data)
        
        # Custom loss with temporal smoothness
        combined_loss, recon_loss, smooth_loss = compute_loss_with_temporal_smoothness(
            recon, neural_data, alpha=0.05
        )
        
        # Total loss with VQ
        loss = combined_loss + vq_weight * vq_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
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
        'train/smooth_loss': total_smooth_loss / num_batches,
        'train/perplexity': total_perplexity / num_batches,
        'train/vq_weight': vq_weight,
        'train/phase': phase
    }

def evaluate_model_single_neuron(model, dataloader, device):
    """Evaluation for single neuron"""
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
    
    # Compute metrics
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    
    # R² score
    y_true = original.flatten()
    y_pred = reconstructed.flatten()
    y_mean = np.mean(y_true)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    
    # Temporal correlation (important for time series)
    correlations = []
    for sample_idx in range(original.shape[0]):
        # Since we have 1 neuron, compute correlation over time
        orig_trace = original[sample_idx, 0, :]  # Shape: (time,)
        recon_trace = reconstructed[sample_idx, 0, :]
        
        if np.std(orig_trace) > 1e-8 and np.std(recon_trace) > 1e-8:
            corr, _ = pearsonr(orig_trace, recon_trace)
            correlations.append(corr if not np.isnan(corr) else 0)
        else:
            correlations.append(0)
    
    # Signal-to-noise ratio
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
    
    return {
        'test/mse': mse,
        'test/mae': mae,
        'test/r_squared': r_squared,
        'test/correlation_mean': np.mean(correlations) if correlations else 0,
        'test/correlation_std': np.std(correlations) if correlations else 0,
        'test/correlation_min': np.min(correlations) if correlations else 0,
        'test/correlation_max': np.max(correlations) if correlations else 1,
        'test/snr_db': snr,
        'test/vq_loss': total_vq_loss / num_batches if num_batches > 0 else 0,
        'test/perplexity': total_perplexity / num_batches if num_batches > 0 else 0,
    }

def visualize_single_neuron_reconstruction(original, reconstructed, epoch):
    """Specialized visualization for single neuron with PEAK-focused selection"""
    
    # ✅ NUOVO: Seleziona i sample con AMPIEZZA MASSIMA
    num_samples = original.shape[0]
    
    # Calcola ampiezza per ogni sample
    amplitudes = []
    for i in range(num_samples):
        trace = original[i, 0, :]  # Single neuron trace
        amplitude = np.max(trace) - np.min(trace)
        amplitudes.append(amplitude)
    
    amplitudes = np.array(amplitudes)
    
    # Seleziona i top 4 sample con ampiezza maggiore
    top_indices = np.argsort(amplitudes)[-4:][::-1]  # Top 4, ordinati dal più alto
    
    print(f"\n  📊 Visualizing samples with HIGHEST amplitude:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"     Rank {rank}: Sample {idx}, Amplitude={amplitudes[idx]:.3f}")
    
    num_examples = len(top_indices)
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(18, 3*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for plot_idx, sample_idx in enumerate(top_indices):
        # Get single neuron trace
        orig_trace = original[sample_idx, 0, :]  # Shape: (time,)
        recon_trace = reconstructed[sample_idx, 0, :]
        
        time_points = np.arange(len(orig_trace))
        
        # Calcola metriche basate su ampiezza/picchi
        amplitude = amplitudes[sample_idx]  # ✅ Usa l'ampiezza già calcolata
        
        # Numero e altezza dei picchi
        mean_val = np.mean(orig_trace)
        std_val = np.std(orig_trace)
        threshold = mean_val + 2 * std_val  # Soglia per identificare picchi
        
        # Identifica picchi
        peak_mask = orig_trace > threshold
        num_peaks = np.sum(np.diff(peak_mask.astype(int)) == 1)  # Conta transizioni 0->1
        
        # Massimo valore assoluto (per catturare eventi estremi)
        max_abs_value = np.max(np.abs(orig_trace))
        
        # Plot 1: Overlay con highlight dei picchi
        axes[plot_idx, 0].plot(time_points, orig_trace, 'b-', label='Original', alpha=0.7, linewidth=1.5)
        axes[plot_idx, 0].plot(time_points, recon_trace, 'r-', label='Reconstruction', alpha=0.7, linewidth=1.5)
        
        # Evidenzia zone dei picchi
        axes[plot_idx, 0].fill_between(time_points, threshold, np.max(orig_trace), 
                                alpha=0.1, color='yellow', label=f'Peak zone (>{threshold:.2f})')
        
        # Marca i picchi individuali
        peak_indices = np.where(peak_mask)[0]
        if len(peak_indices) > 0:
            axes[plot_idx, 0].scatter(peak_indices, orig_trace[peak_indices], 
                             color='blue', s=30, zorder=5, alpha=0.8, marker='^', label='Peaks (orig)')
            axes[plot_idx, 0].scatter(peak_indices, recon_trace[peak_indices], 
                             color='red', s=30, zorder=5, alpha=0.8, marker='v', label='Peaks (recon)')
        
        axes[plot_idx, 0].set_xlabel('Time')
        axes[plot_idx, 0].set_ylabel('Activity')
        axes[plot_idx, 0].set_title(f'Rank {plot_idx+1} (Sample {sample_idx}): Peaks={num_peaks}, Amp={amplitude:.2f}, Max={max_abs_value:.2f}')
        axes[plot_idx, 0].legend(loc='upper right', fontsize=8)
        axes[plot_idx, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error con focus sui picchi
        error = np.abs(orig_trace - recon_trace)
        axes[plot_idx, 1].plot(time_points, error, 'k-', alpha=0.7, linewidth=1)
        axes[plot_idx, 1].fill_between(time_points, 0, error, alpha=0.3, color='red')
        
        # Evidenzia errore nelle zone dei picchi
        peak_error = error.copy()
        peak_error[~peak_mask] = 0
        axes[plot_idx, 1].fill_between(time_points, 0, peak_error, alpha=0.6, color='darkred', 
                               label=f'Peak error')
        
        axes[plot_idx, 1].set_xlabel('Time')
        axes[plot_idx, 1].set_ylabel('Absolute Error')
        axes[plot_idx, 1].set_title(f'Rank {plot_idx+1}: Error (Peak err mean: {np.mean(error[peak_mask]):.3f})' if np.sum(peak_mask) > 0 else f'Rank {plot_idx+1}: Error (No peaks)')
        axes[plot_idx, 1].legend(loc='upper right', fontsize=8)
        axes[plot_idx, 1].grid(True, alpha=0.3)
        
        # Plot 3: Zoom sui picchi più grandi
        axes[plot_idx, 2].set_title(f'Rank {plot_idx+1}: Peak Reconstruction Quality')
        
        if len(peak_indices) > 0:
            # Trova i 3 picchi più alti
            peak_heights = orig_trace[peak_indices]
            top_peak_idx = peak_indices[np.argsort(peak_heights)[-min(3, len(peak_indices)):]]
            
            # Crea zoom windows attorno ai top picchi
            window_size = 5  # ±5 timesteps attorno al picco
            
            for j, peak_idx in enumerate(top_peak_idx):
                start = max(0, peak_idx - window_size)
                end = min(len(orig_trace), peak_idx + window_size + 1)
                
                local_time = np.arange(start, end)
                
                # Offset verticale per separare i picchi nel plot
                offset = j * (amplitude * 0.4)
                
                axes[plot_idx, 2].plot(local_time - peak_idx, orig_trace[start:end] + offset, 
                              'b.-', alpha=0.7, label=f'Peak {j+1} orig' if j == 0 else "")
                axes[plot_idx, 2].plot(local_time - peak_idx, recon_trace[start:end] + offset, 
                              'r.-', alpha=0.7, label=f'Peak {j+1} recon' if j == 0 else "")
                
                # Linea verticale al centro del picco
                axes[plot_idx, 2].axvline(0, color='gray', linestyle='--', alpha=0.3)
                
                # Annotazione con l'errore del picco
                peak_error_val = np.abs(orig_trace[peak_idx] - recon_trace[peak_idx])
                axes[plot_idx, 2].text(0.5, orig_trace[peak_idx] + offset, 
                              f'Err: {peak_error_val:.3f}',
                              fontsize=8, ha='left', va='bottom')
            
            axes[plot_idx, 2].set_xlabel('Time (relative to peak)')
            axes[plot_idx, 2].set_ylabel('Activity (stacked)')
            axes[plot_idx, 2].legend(loc='upper right', fontsize=8)
        else:
            # Se non ci sono picchi significativi, mostra istogramma dei valori
            axes[plot_idx, 2].hist(orig_trace, bins=20, alpha=0.5, color='blue', label='Original', density=True)
            axes[plot_idx, 2].hist(recon_trace, bins=20, alpha=0.5, color='red', label='Reconstruction', density=True)
            axes[plot_idx, 2].set_xlabel('Activity Value')
            axes[plot_idx, 2].set_ylabel('Density')
            axes[plot_idx, 2].legend()
            axes[plot_idx, 2].set_title(f'Rank {plot_idx+1}: No significant peaks - Value distribution')
        
        axes[plot_idx, 2].grid(True, alpha=0.3)
        
        # Aggiungi metriche dettagliate sui picchi
        if np.std(orig_trace) > 1e-8 and np.std(recon_trace) > 1e-8:
            corr, _ = pearsonr(orig_trace, recon_trace)
            
            # Correlazione solo sui picchi (se esistono abbastanza punti)
            if np.sum(peak_mask) >= 2:  # pearsonr needs at least 2 points
                peak_corr, _ = pearsonr(orig_trace[peak_mask], recon_trace[peak_mask])
            else:
                peak_corr = np.nan
            
            # Box con statistiche
            stats_text = f'Overall Corr: {corr:.3f}\n'
            if not np.isnan(peak_corr):
                stats_text += f'Peak Corr: {peak_corr:.3f}\n'
            elif np.sum(peak_mask) == 1:
                stats_text += f'Peak Corr: N/A (1 point)\n'
            else:
                stats_text += f'Peak Corr: N/A (no peaks)\n'
            stats_text += f'Amplitude: {amplitude:.2f}\n'
            stats_text += f'Num Peaks: {num_peaks}\n'
            stats_text += f'Max |value|: {max_abs_value:.2f}'
            
            axes[plot_idx, 2].text(0.02, 0.98, stats_text, 
                          transform=axes[plot_idx, 2].transAxes, 
                          verticalalignment='top',
                          fontsize=8,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Single Neuron Reconstructions (TOP AMPLITUDE Samples) - Epoch {epoch}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to numpy array first, then to PIL Image
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Create PIL image from numpy array
    pil_image = Image.fromarray(buf)
    
    return pil_image


def compute_peak_focused_metrics(original, reconstructed):
    """
    Compute metrics specifically focused on peak reconstruction quality
    """
    metrics = {}
    
    # Flatten for overall metrics
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    # Overall metrics
    metrics['mse_overall'] = np.mean((orig_flat - recon_flat) ** 2)
    metrics['mae_overall'] = np.mean(np.abs(orig_flat - recon_flat))
    
    # Peak-focused metrics
    peak_metrics = []
    amplitude_metrics = []
    
    for sample_idx in range(original.shape[0]):
        orig_trace = original[sample_idx, 0, :]
        recon_trace = reconstructed[sample_idx, 0, :]
        
        # Calculate amplitude
        amplitude_orig = np.max(orig_trace) - np.min(orig_trace)
        amplitude_recon = np.max(recon_trace) - np.min(recon_trace)
        amplitude_error = np.abs(amplitude_orig - amplitude_recon)
        amplitude_metrics.append(amplitude_error / (amplitude_orig + 1e-8))
        
        # Identify peaks (2 sigma above mean)
        mean_val = np.mean(orig_trace)
        std_val = np.std(orig_trace)
        threshold = mean_val + 2 * std_val
        
        peak_mask = orig_trace > threshold
        
        if np.sum(peak_mask) > 0:
            # MSE only on peaks
            peak_mse = np.mean((orig_trace[peak_mask] - recon_trace[peak_mask]) ** 2)
            peak_metrics.append(peak_mse)
            
            # Check if peaks are preserved in reconstruction
            recon_peaks = recon_trace > threshold
            peak_preservation = np.sum(peak_mask & recon_peaks) / np.sum(peak_mask)
            metrics[f'sample_{sample_idx}_peak_preservation'] = peak_preservation
    
    # Aggregate peak metrics
    if peak_metrics:
        metrics['mse_peaks'] = np.mean(peak_metrics)
        metrics['mse_peaks_std'] = np.std(peak_metrics)
    else:
        metrics['mse_peaks'] = np.nan
        metrics['mse_peaks_std'] = np.nan
    
    metrics['amplitude_error_mean'] = np.mean(amplitude_metrics)
    metrics['amplitude_error_std'] = np.std(amplitude_metrics)
    
    return metrics
def main():
    # Initialize W&B
    wandb.init(
        project="calcium-vqvae-single-neuron",
        name="single-neuron-fixed-architecture",
        config={
            "model_type": "CalciumVQVAE_SingleNeuron",
            "num_sessions_train": len(TRAINING_SESSION_IDS),
            "num_sessions_test": len(TEST_SESSION_IDS),
            "window_size": 60,
            "stride": 50,
            
            # Training parameters
            "batch_size": 64,  # Larger batch for single neuron
            "learning_rate": 0.0005,  # Lower LR for stability
            "max_epochs": 205,
            "patience": 40,
            
            # VQ scheduling
            "warmup_epochs": 30,
            
            **MODEL_CONFIG
        }
    )
    
    print("🧠 SINGLE NEURON VQ-VAE TRAINING (FIXED)")
    print("="*70)
    print(f"✅ Model config:")
    for key, value in MODEL_CONFIG.items():
        print(f"   {key}: {value}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"💻 Using device: {device}")
    
    try:
        # Load datasets
        print("\n📊 Loading datasets...")
        train_loader, test_loader, dataset_info = create_unified_dataloaders(
            train_session_ids=TRAINING_SESSION_IDS,
            batch_size=wandb.config.batch_size,
            test_split=0.2,
            window_size=wandb.config.window_size,
            stride=wandb.config.stride,
            min_neurons=wandb.config.num_neurons,  # 1 neuron
            num_workers=0,
            use_cross_session_test=False
        )
        
        print(f"✅ Dataset loaded:")
        print(f"   Sessions: {dataset_info['num_sessions_train']}")
        print(f"   Train samples: {dataset_info['train_samples']}")
        print(f"   Test samples: {dataset_info['test_samples']}")
        
        # Create model
        print("\n🏗️ Creating model...")
        model = create_single_neuron_vqvae().to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        wandb.log({"model/num_parameters": num_params})
        print(f"✅ Model created: {num_params:,} parameters")
        
        # Optimizer with cosine annealing
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
        best_train_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(wandb.config.max_epochs):
            # Train
            train_metrics = train_epoch_single_neuron(
                model, train_loader, optimizer, device, 
                epoch, wandb.config
            )
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Evaluate every 5 epochs
            if epoch % 5 == 0:
                test_metrics = evaluate_model_single_neuron(model, test_loader, device)
                
                # Log all metrics
                all_metrics = {
                    **train_metrics, 
                    **test_metrics, 
                    'epoch': epoch,
                    'learning_rate': current_lr
                }
                wandb.log(all_metrics)
                
                # Print progress
                print(f"Epoch {epoch:3d} [{train_metrics['train/phase']}]:")
                print(f"  Train Loss: {train_metrics['train/recon_loss']:.6f}")
                print(f"  Test MSE: {test_metrics['test/mse']:.6f}")
                print(f"  Test Corr: {test_metrics['test/correlation_mean']:.4f} ± {test_metrics['test/correlation_std']:.4f}")
                print(f"  R²: {test_metrics['test/r_squared']:.4f}, SNR: {test_metrics['test/snr_db']:.1f} dB")
                print(f"  Perplexity: {train_metrics['train/perplexity']:.1f}")
                
                # Visualize reconstructions
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
                
                # Check for improvement
                current_test_corr = test_metrics['test/correlation_mean']
                if current_test_corr > best_test_corr:
                    best_test_corr = current_test_corr
                    patience_counter = 0
                    
                    # Save best model
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_config': MODEL_CONFIG,
                        'best_correlation': best_test_corr,
                        'test_metrics': test_metrics
                    }
                    torch.save(checkpoint, './results/best_single_neuron_model.pth')
                    print(f"  💾 Saved best model (corr={best_test_corr:.4f})")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= wandb.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                wandb.log({**train_metrics, 'epoch': epoch, 'learning_rate': current_lr})
        
        # Final evaluation
        print("\n🎯 FINAL EVALUATION:")
        final_metrics = evaluate_model_single_neuron(model, test_loader, device)
        
        print(f"Test MSE: {final_metrics['test/mse']:.6f}")
        print(f"Test Correlation: {final_metrics['test/correlation_mean']:.4f}")
        print(f"R²: {final_metrics['test/r_squared']:.4f}")
        print(f"SNR: {final_metrics['test/snr_db']:.1f} dB")
        
        wandb.log({
            'final/test_mse': final_metrics['test/mse'],
            'final/correlation': final_metrics['test/correlation_mean'],
            'final/r_squared': final_metrics['test/r_squared'],
            'final/snr_db': final_metrics['test/snr_db'],
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