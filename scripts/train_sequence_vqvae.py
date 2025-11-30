#!/usr/bin/env python3
"""
VQ-VAE Training for Sequence Behavioral Prediction - ALL NEURONS VERSION

Combina:
- Architettura VQ-VAE (come train_di_1_neurone.py) 
- Task di predizione sequenze temporali (come train_sequence_behavioral.py)
- USA TUTTI I NEURONI della sessione (non solo 1!)

Differenze chiave da train_di_1_neurone.py:
✅ num_neurons = TUTTI i neuroni della sessione (auto-detected)
✅ Output decoder: 1 channel (velocity sequence) invece di ricostruire neuroni
✅ Loss: sequence MSE con optional peak weighting
✅ Metriche: correlazione temporale su sequenze di velocità

Differenze da train_sequence_behavioral.py:
✅ Usa VQ-VAE invece di SequenceBehavioralRegressor
✅ Include quantization loss e codebook
✅ Encoder/decoder più potenti con residual blocks
"""

import sys
import os
from pathlib import Path

# Aggiungi root al PYTHONPATH
root_dir = Path(__file__).parent.parent if Path(__file__).parent.name == 'scripts' else Path(__file__).parent
sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from models.vqvae import CalciumVQVAE
from datasets.sequence_behavioral import create_sequence_dataloaders
from datasets.calcium import TRAINING_SESSION_IDS


# ============================================================================
# MODEL CONFIGURATION - Ottimizzato per TUTTI i neuroni + sequence prediction
# ============================================================================

MODEL_CONFIG = {
    'num_neurons': None,        # ✅ Determinato automaticamente dal dataset
    'num_hiddens': 128,         # Hidden dimension aumentata
    'num_residual_layers': 3,   # Più layer per catturare pattern temporali
    'num_residual_hiddens': 64, 
    'num_embeddings': 512,     # Codebook grande per varietà sequenze
    'embedding_dim': 128,       # Embeddings ricchi
    'commitment_cost': 0.5,     # Bilanciato
    'dropout_rate': 0.1,        # Dropout leggero
    'use_quantizer': True,
}


# ============================================================================
# CUSTOM DECODER FOR VELOCITY SEQUENCES
# ============================================================================

class VelocitySequenceDecoder(nn.Module):
    """
    Decoder specializzato per sequenze temporali di velocità
    
    Input: quantized features (B, embedding_dim, compressed_time)
    Output: velocity sequence (B, 1, 60)
    """
    
    def __init__(self, embedding_dim, num_hiddens, num_residual_layers, 
                 num_residual_hiddens, dropout_rate=0.3):
        super(VelocitySequenceDecoder, self).__init__()
        
        from models.encoder import ImprovedResidualBlock
        
        # Residual stack
        self._residual_stack = nn.ModuleList([
            ImprovedResidualBlock(embedding_dim, embedding_dim, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])
        
        # Progressive upsampling - mantiene risoluzione temporale
        self._conv_transpose_1 = nn.ConvTranspose1d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        
        self._conv_transpose_2 = nn.ConvTranspose1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens//2,
            kernel_size=5, stride=2, padding=2, output_padding=0
        )
        
        # Final projection - OUTPUT: 1 channel (velocity)
        self._conv_final = nn.Conv1d(
            in_channels=num_hiddens//2,
            out_channels=1,  # ✅ 1 channel per velocity!
            kernel_size=7, stride=1, padding=3
        )
        
        self._dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Residual processing
        for block in self._residual_stack:
            x = block(x)
        
        # Upsampling
        x = F.relu(self._conv_transpose_1(x))
        x = self._dropout(x)
        x = F.relu(self._conv_transpose_2(x))
        x = self._dropout(x)
        
        # Final projection to velocity
        x = self._conv_final(x)
        
        return x


class VelocityVQVAE(nn.Module):
    """
    VQ-VAE ottimizzato per predizione di sequenze di velocità
    
    ✅ KEY DIFFERENCE: Usa TUTTI i neuroni come input
    
    Differenze dal CalciumVQVAE standard:
    - Input: TUTTI i neuroni della sessione (non solo 1)
    - Decoder specializzato per output 1 channel (velocity sequence)
    - Loss pesata su picchi (opzionale)
    - Metriche su correlazione temporale
    """
    
    def __init__(self, num_neurons, num_hiddens=128, num_residual_layers=3, 
                 num_residual_hiddens=64, num_embeddings=1024, embedding_dim=128, 
                 commitment_cost=0.5, dropout_rate=0.1, use_quantizer=True):
        super(VelocityVQVAE, self).__init__()
        
        self.use_quantizer = use_quantizer
        self.num_neurons = num_neurons
        
        # ✅ ENCODER: Processa TUTTI i neuroni
        from models.encoder import CalciumEncoder
        
        self.encoder = CalciumEncoder(
            num_neurons,  # ✅ TUTTI i neuroni!
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            dropout_rate=dropout_rate
        )
        
        # Pre-quantization conv
        self.pre_quantization_conv = nn.Conv1d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1
        )
        
        # Vector Quantizer
        from models.quantizer import ImprovedVectorQuantizer
        
        self.vector_quantization = ImprovedVectorQuantizer(
            num_embeddings,
            embedding_dim,
            commitment_cost,
            decay=0.99,
            eps=1e-5
        )
        
        # ✅ VELOCITY SEQUENCE DECODER (specializzato!)
        self.decoder = VelocitySequenceDecoder(
            embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            dropout_rate
        )
        
        print(f"✅ VelocityVQVAE creato:")
        print(f"   Input: {num_neurons} neuroni")
        print(f"   Embeddings: {num_embeddings}")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Output: 1 channel (velocity sequence)")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (B, num_neurons, 60) neural data - TUTTI I NEURONI!
        
        Returns:
            vq_loss, velocity_seq, perplexity, quantized, encodings
            velocity_seq shape: (B, 1, 60) - velocity sequence!
        """
        # Encode - processa TUTTI i neuroni insieme
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        
        # Normalize and clip
        z = F.layer_norm(z, [z.size(1), z.size(2)])
        z = torch.clamp(z, min=-10.0, max=10.0)
        
        if self.use_quantizer:
            vq_loss, quantized, perplexity, encodings, encoding_indices = \
                self.vector_quantization(z)
        else:
            quantized = z
            vq_loss = torch.tensor(0.0, device=z.device)
            perplexity = torch.tensor(float(self.vector_quantization.n_e), device=z.device)
            encodings = None
            encoding_indices = None
        
        # Decode to velocity sequence
        velocity_seq = self.decoder(quantized)  # (B, 1, 60)
        
        # Match dimensioni temporali se necessario
        if velocity_seq.shape[2] != x.shape[2]:
            velocity_seq = F.interpolate(
                velocity_seq, size=x.shape[2], mode='linear', align_corners=False
            )
        
        return vq_loss, velocity_seq, perplexity, quantized, encodings
    
    def encode(self, x):
        """Encode to quantized representation"""
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        
        if self.use_quantizer:
            _, quantized, _, _, _ = self.vector_quantization(z)
        else:
            quantized = z
        
        return quantized
    
    def decode(self, quantized):
        """Decode to velocity sequence"""
        return self.decoder(quantized)
    
    def get_codebook_usage(self):
        """Get codebook usage statistics"""
        if hasattr(self.vector_quantization, 'get_usage_stats'):
            return self.vector_quantization.get_usage_stats()
        else:
            return {'usage_percentage': 100.0}


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def weighted_sequence_loss(predictions, targets, threshold=1.5, weight_high=3.0):
    """
    MSE pesata che dà più importanza ai timesteps con picchi
    
    Args:
        predictions: (batch, 1, 60) o (batch, 60) sequenze predette
        targets: (batch, 60) sequenze target
    
    Returns:
        weighted loss
    """
    # Squeeze predictions se necessario
    if predictions.dim() == 3:
        predictions = predictions.squeeze(1)  # (batch, 60)
    
    # Errori per timestep
    errors = (predictions - targets) ** 2
    
    # Identifica timesteps con picchi
    is_peak = torch.abs(targets) > threshold
    
    # Pesi
    weights = torch.where(is_peak,
                         torch.tensor(weight_high, device=targets.device),
                         torch.tensor(1.0, device=targets.device))
    
    # Loss pesata
    weighted_loss = torch.mean(weights * errors)
    
    return weighted_loss


def train_epoch(model, dataloader, optimizer, device, epoch, config):
    """
    Training epoch con VQ-VAE per velocity prediction
    """
    model.train()
    
    # VQ Weight Scheduling
    warmup_epochs = config.get('warmup_epochs', 30)
    max_epochs = config.get('max_epochs', 200)
    
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
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{phase}]")
    
    for batch_idx, (neural_data, velocity_seq) in enumerate(pbar):
        neural_data = neural_data.to(device)  # (B, ALL_neurons, 60)
        velocity_seq = velocity_seq.to(device)  # (B, 60)
        
        # Forward pass
        vq_loss, velocity_pred, perplexity, _, _ = model(neural_data)
        # velocity_pred: (B, 1, 60)
        
        # Squeeze predictions
        velocity_pred = velocity_pred.squeeze(1)  # (B, 60)
        
        # Reconstruction loss (weighted se abilitato)
        if config.get('use_weighted_loss', False):
            recon_loss = weighted_sequence_loss(
                velocity_pred, velocity_seq,
                threshold=config.get('peak_threshold', 1.5),
                weight_high=config.get('peak_weight', 3.0)
            )
        else:
            recon_loss = F.mse_loss(velocity_pred, velocity_seq)
        
        # Total loss
        loss = recon_loss + vq_weight * vq_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'recon': f'{recon_loss.item():.6f}',
            'vq': f'{vq_loss.item():.6f}'
        })
    
    num_batches = len(dataloader)
    return {
        'train/total_loss': total_loss / num_batches,
        'train/recon_loss': total_recon_loss / num_batches,
        'train/vq_loss': total_vq_loss / num_batches,
        'train/perplexity': total_perplexity / num_batches,
        'train/vq_weight': vq_weight,
        'train/phase': phase
    }


def evaluate(model, dataloader, device):
    """
    Evaluation con metriche su sequenze
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_vq_loss = 0
    total_perplexity = 0
    num_batches = 0
    
    with torch.no_grad():
        for neural_data, velocity_seq in tqdm(dataloader, desc="Evaluating"):
            neural_data = neural_data.to(device)
            velocity_seq = velocity_seq.to(device)
            
            # Forward
            vq_loss, velocity_pred, perplexity, _, _ = model(neural_data)
            velocity_pred = velocity_pred.squeeze(1)
            
            all_predictions.append(velocity_pred.cpu().numpy())
            all_targets.append(velocity_seq.cpu().numpy())
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
    
    # Concatenate
    predictions = np.concatenate(all_predictions, axis=0)  # (N, 60)
    targets = np.concatenate(all_targets, axis=0)  # (N, 60)
    
    # Compute metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    # Global correlation (flatten tutto)
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    global_corr = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # Correlation per sample (media su timesteps)
    correlations_per_sample = []
    for i in range(len(predictions)):
        if np.std(predictions[i]) > 1e-8 and np.std(targets[i]) > 1e-8:
            corr, _ = pearsonr(predictions[i], targets[i])
            if not np.isnan(corr):
                correlations_per_sample.append(corr)
    
    avg_corr_per_sample = np.mean(correlations_per_sample) if correlations_per_sample else 0
    
    # Correlation per timestep
    correlations_per_timestep = []
    for t in range(60):
        if np.std(predictions[:, t]) > 1e-8 and np.std(targets[:, t]) > 1e-8:
            corr = np.corrcoef(predictions[:, t], targets[:, t])[0, 1]
            if not np.isnan(corr):
                correlations_per_timestep.append(corr)
            else:
                correlations_per_timestep.append(0)
        else:
            correlations_per_timestep.append(0)
    
    avg_corr_per_timestep = np.mean(correlations_per_timestep)
    
    # Peak metrics
    peak_mask = np.abs(targets) > 1.5
    if np.sum(peak_mask) > 0:
        peak_mse = np.mean((predictions[peak_mask] - targets[peak_mask]) ** 2)
        peak_mae = np.mean(np.abs(predictions[peak_mask] - targets[peak_mask]))
    else:
        peak_mse = np.nan
        peak_mae = np.nan
    
    return {
        'test/mse': mse,
        'test/mae': mae,
        'test/global_correlation': global_corr,
        'test/avg_corr_per_sample': avg_corr_per_sample,
        'test/avg_corr_per_timestep': avg_corr_per_timestep,
        'test/peak_mse': peak_mse if not np.isnan(peak_mse) else 0,
        'test/peak_mae': peak_mae if not np.isnan(peak_mae) else 0,
        'test/vq_loss': total_vq_loss / num_batches if num_batches > 0 else 0,
        'test/perplexity': total_perplexity / num_batches if num_batches > 0 else 0,
    }, predictions, targets, correlations_per_timestep


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_sequence_examples(predictions, targets, epoch, n_samples=6):
    """Plotta esempi di sequenze ricostruite"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Seleziona sample con maggiore ampiezza
    amplitudes = [np.max(targets[i]) - np.min(targets[i]) for i in range(len(targets))]
    top_indices = np.argsort(amplitudes)[-n_samples:][::-1]
    
    for plot_idx, idx in enumerate(top_indices):
        ax = axes[plot_idx]
        
        pred_seq = predictions[idx]
        target_seq = targets[idx]
        
        timesteps = np.arange(60)
        
        # Plot
        ax.plot(timesteps, target_seq, 'b-', linewidth=2, label='Target', alpha=0.8)
        ax.plot(timesteps, pred_seq, 'r--', linewidth=2, label='Prediction', alpha=0.8)
        ax.fill_between(timesteps, target_seq, pred_seq, alpha=0.2, color='gray')
        
        # Calcola metriche
        corr = np.corrcoef(pred_seq, target_seq)[0, 1]
        mse = np.mean((pred_seq - target_seq) ** 2)
        amplitude = amplitudes[idx]
        
        # Identifica picchi
        threshold = np.mean(target_seq) + 2 * np.std(target_seq)
        has_peaks = np.any(target_seq > threshold)
        
        title = f'Sample {idx}: r={corr:.3f}, MSE={mse:.4f}, Amp={amplitude:.2f}'
        if has_peaks:
            title += ' [PEAKS]'
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=9)
        ax.set_ylabel('Velocity (normalized)', fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Epoch {epoch}: VQ-VAE Velocity Sequence Reconstruction (ALL NEURONS)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_correlation_per_timestep(correlations, epoch):
    """Plotta correlazione per timestep"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    timesteps = np.arange(60)
    mean_corr = np.mean(correlations)
    
    ax.plot(timesteps, correlations, 'b-o', linewidth=2, markersize=4)
    ax.axhline(y=mean_corr, color='r', linestyle='--', 
               linewidth=2, label=f'Mean: {mean_corr:.3f}')
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title(f'Epoch {epoch}: Correlation per Timestep (VQ-VAE, ALL NEURONS)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Config
    config = {
        # Data
        'session_id': TRAINING_SESSION_IDS[2],
        'window_size': 60,
        'stride': 30,
        'test_split': 0.2,
        'normalize': True,
        
        # Training
        'batch_size': 32,
        'num_epochs': 200,
        'learning_rate': 5e-4,
        'weight_decay': 1e-5,
        
        # VQ scheduling
        'warmup_epochs': 30,
        'max_epochs': 200,
        
        # Loss
        'use_weighted_loss': False,
        'peak_threshold': 1.5,
        'peak_weight': 3.0,
        
        # Paths
        'save_dir': './results/velocity_vqvae_all_neurons',
        
        # System
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 0,
        'seed': 42,
        
        **MODEL_CONFIG
    }
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Save directory: {save_dir}")
    
    device = torch.device(config['device'])
    print(f"🖥️  Device: {device}")
    
    # Initialize W&B
    wandb.init(
        project="calcium-vqvae-velocity-sequences",
        name=f"velocity_{MODEL_CONFIG['num_embeddings']}_{TRAINING_SESSION_IDS[2]}",
        config=config,
        tags=['vqvae', 'velocity_prediction', 'all_neurons', 'temporal']
    )
    
    print(f"✅ W&B initialized: {wandb.run.name}")
    
    # Load data
    print("\n" + "="*70)
    print("📊 LOADING DATA")
    print("="*70)
    
    train_loader, test_loader, dataset_info = create_sequence_dataloaders(
        session_ids=[config['session_id']],
        batch_size=config['batch_size'],
        test_split=config['test_split'],
        window_size=config['window_size'],
        stride=config['stride'],
        num_workers=config['num_workers'],
        normalize=config['normalize']
    )
    
    print(f"\n✅ Data loaded:")
    print(f"   Train: {dataset_info['train_samples']}")
    print(f"   Test: {dataset_info['test_samples']}")
    print(f"   Neural shape: {dataset_info['neural_shape']}")
    print(f"   Velocity shape: {dataset_info['velocity_shape']}")
    
    # ✅ Determine num_neurons from dataset
    sample_batch = next(iter(train_loader))
    sample_neural = sample_batch[0]  # (batch, neurons, time)
    num_neurons = sample_neural.shape[1]
    
    print(f"\n📊 Detected from dataset:")
    print(f"   Number of neurons: {num_neurons} ← ALL NEURONS!")
    
    # Update config
    MODEL_CONFIG['num_neurons'] = num_neurons
    config['num_neurons'] = num_neurons
    
    # Create model
    print("\n" + "="*70)
    print("🧠 CREATING VQ-VAE MODEL (ALL NEURONS)")
    print("="*70)
    
    model = VelocityVQVAE(
        num_neurons=num_neurons,  # ✅ ALL NEURONS!
        num_hiddens=MODEL_CONFIG['num_hiddens'],
        num_residual_layers=MODEL_CONFIG['num_residual_layers'],
        num_residual_hiddens=MODEL_CONFIG['num_residual_hiddens'],
        num_embeddings=MODEL_CONFIG['num_embeddings'],
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        commitment_cost=MODEL_CONFIG['commitment_cost'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        use_quantizer=MODEL_CONFIG['use_quantizer']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model created: {total_params:,} parameters")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )
    
    print(f"✅ Optimizer: AdamW (lr={config['learning_rate']})")
    print(f"✅ Scheduler: CosineAnnealingLR")
    
    # Training loop
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    
    best_correlation = -np.inf
    best_epoch = 0
    patience = 40
    patience_counter = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, config)
        
        # Update LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            test_metrics, predictions, targets, corrs_per_timestep = evaluate(model, test_loader, device)
            
            # Log metrics
            all_metrics = {
                **train_metrics,
                **test_metrics,
                'epoch': epoch,
                'learning_rate': current_lr
            }
            wandb.log(all_metrics)
            
            # Print
            print(f"\n📊 Epoch {epoch} Results [{train_metrics['train/phase']}]:")
            print(f"   Train Loss: {train_metrics['train/recon_loss']:.6f}")
            print(f"   Test MSE: {test_metrics['test/mse']:.6f}")
            print(f"   Global Corr: {test_metrics['test/global_correlation']:.4f}")
            print(f"   Avg Corr/Sample: {test_metrics['test/avg_corr_per_sample']:.4f}")
            print(f"   Avg Corr/Timestep: {test_metrics['test/avg_corr_per_timestep']:.4f}")
            print(f"   Perplexity: {train_metrics['train/perplexity']:.1f}")
            
            # Visualize
            if epoch % 20 == 0:
                fig_examples = plot_sequence_examples(predictions, targets, epoch, n_samples=6)
                fig_corr = plot_correlation_per_timestep(corrs_per_timestep, epoch)
                
                wandb.log({
                    'sequence_examples': wandb.Image(fig_examples),
                    'correlation_per_timestep': wandb.Image(fig_corr),
                    'epoch': epoch
                })
                
                plt.close(fig_examples)
                plt.close(fig_corr)
            
            # Save best model
            current_corr = test_metrics['test/global_correlation']
            if current_corr > best_correlation:
                best_correlation = current_corr
                best_epoch = epoch
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_config': MODEL_CONFIG,
                    'best_correlation': best_correlation,
                    'test_metrics': test_metrics,
                    'num_neurons': num_neurons
                }
                
                checkpoint_path = save_dir / f'velocity_{MODEL_CONFIG["num_embeddings"]}_{TRAINING_SESSION_IDS[2]}.pth'
                torch.save(checkpoint, checkpoint_path)
                
                print(f"\n💾 Saved best model (corr={best_correlation:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        else:
            wandb.log({**train_metrics, 'epoch': epoch, 'learning_rate': current_lr})
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE")
    print("="*70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best correlation: {best_correlation:.4f}")
    print(f"Number of neurons used: {num_neurons} (ALL NEURONS!)")
    
    wandb.run.summary['best_epoch'] = best_epoch
    wandb.run.summary['best_correlation'] = best_correlation
    wandb.run.summary['num_neurons'] = num_neurons
    
    wandb.finish()
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()