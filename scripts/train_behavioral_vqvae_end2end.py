#!/usr/bin/env python3
"""
End-to-End Training for Behavioral VQ-VAE

Training completo end-to-end (come singolo neurone) ma per predizione velocità.
- Encoder + Codebook + Decoder: TUTTI TRAINABLE
- Input: tutti i neuroni (num_neurons, 1, 60)
- Output: velocità (60 timesteps)
- Nessun pretrained, tutto da zero
"""

import sys
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent
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
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    'num_neurons': None,        # Determinato automaticamente dal dataset
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 1024,
    'embedding_dim': 128,
    'commitment_cost': 0.5,
    'dropout_rate': 0.0,
    'use_quantizer': True,
}


# ============================================================================
# BEHAVIORAL DECODER (sostituisce decoder standard)
# ============================================================================

class BehavioralVelocityDecoder(nn.Module):
    """
    Decoder per predire sequenze di velocità invece di ricostruire neuroni
    
    Input: quantized features (batch, embedding_dim, 15) - aggregato da encoder
    Output: velocity sequence (batch, 60)
    """
    
    def __init__(self, embedding_dim, num_hiddens=128, 
                 num_residual_layers=3, num_residual_hiddens=64, dropout_rate=0.1):
        super(BehavioralVelocityDecoder, self).__init__()
        
        # Input: (batch, embedding_dim, 15) - già aggregato dall'encoder
        input_channels = embedding_dim
        
        print(f"\n🔨 Building Behavioral Velocity Decoder:")
        print(f"   Input: {embedding_dim} channels (aggregated from encoder)")
        
        # Stage 1: Channel reduction (aumentata capacità)
        self.reduction = nn.Sequential(
            nn.Conv1d(input_channels, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Stage 2: Temporal upsampling 15 → 30 → 60
        self.upsample = nn.Sequential(
            # 15 → 30
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 30 → 60
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Stage 3: Refinement
        self.refinement = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Stage 4: Output (velocity)
        self.output = nn.Conv1d(32, 1, kernel_size=1)
        
        # Initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Architecture: {input_channels} → 512 → 256 → [upsample] → 64 → 32 → 1")
        print(f"   Total params: {total_params:,}")
    
    def forward(self, x):
        """
        Args:
            x: (batch, embedding_dim, 15) quantized features (already aggregated)
        
        Returns:
            velocity: (batch, 60) velocity sequence
        """
        # Handle single sample
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (embedding_dim, 15) -> (1, embedding_dim, 15)
        
        # Reduce channels
        x = self.reduction(x)  # (batch, 256, 15)
        
        # Temporal upsampling
        x = self.upsample(x)  # (batch, 64, 60)
        
        # Refinement
        x = self.refinement(x)  # (batch, 32, 60)
        
        # Output
        x = self.output(x)  # (batch, 1, 60)
        
        # Squeeze channel dimension
        x = x.squeeze(1)  # (batch, 60)
        
        return x


# ============================================================================
# BEHAVIORAL VQ-VAE (End-to-End)
# ============================================================================

class BehavioralVQVAEEnd2End(nn.Module):
    """
    VQ-VAE end-to-end per predizione velocità comportamentale
    
    - Encoder: trainable
    - Codebook: trainable
    - Decoder: trainable (specializzato per velocità)
    """
    
    def __init__(self, num_neurons, num_hiddens=128, num_residual_layers=3,
                 num_residual_hiddens=64, num_embeddings=1024, embedding_dim=128,
                 commitment_cost=0.5, dropout_rate=0.0, use_quantizer=True):
        super(BehavioralVQVAEEnd2End, self).__init__()
        
        self.num_neurons = num_neurons
        self.use_quantizer = use_quantizer
        
        # Encoder (trainable)
        from models.encoder import CalciumEncoder
        self.encoder = CalciumEncoder(
            num_neurons,
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
        
        # Vector Quantizer (trainable)
        from models.quantizer import ImprovedVectorQuantizer
        self.vector_quantization = ImprovedVectorQuantizer(
            num_embeddings,
            embedding_dim,
            commitment_cost,
            decay=0.99,
            eps=1e-5
        )
        
        # Behavioral Decoder (trainable)
        self.decoder = BehavioralVelocityDecoder(
            embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            dropout_rate
        )
        
        print(f"\n✅ BehavioralVQVAE End-to-End created:")
        print(f"   Neurons: {num_neurons}")
        print(f"   Embeddings: {num_embeddings}")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   All components: TRAINABLE")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch, num_neurons, 60) or (num_neurons, 60) neural data
        
        Returns:
            vq_loss, velocity_pred, perplexity, quantized, encodings
            velocity_pred: (batch, 60) or (60,) velocity sequence
        """
        # Handle single sample
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (num_neurons, 60) -> (1, num_neurons, 60)
            squeeze_output = True
        
        # Encode: expects (batch, num_neurons, 60)
        z = self.encoder(x)  # (batch, num_hiddens, 15)
        z = self.pre_quantization_conv(z)  # (batch, embedding_dim, 15)
        
        # Normalize and clip
        z = F.layer_norm(z, [z.size(1), z.size(2)])
        z = torch.clamp(z, min=-10.0, max=10.0)
        
        # Quantize
        if self.use_quantizer:
            vq_loss, quantized, perplexity, encodings, encoding_indices = \
                self.vector_quantization(z)
        else:
            quantized = z
            vq_loss = torch.tensor(0.0, device=z.device)
            perplexity = torch.tensor(float(self.vector_quantization.n_e), device=z.device)
            encodings = None
            encoding_indices = None
        
        # Decode to velocity (process entire batch at once)
        velocity_pred = self.decoder(quantized)  # (batch, 60)
        
        if squeeze_output:
            velocity_pred = velocity_pred.squeeze(0)  # (60,)
        
        return vq_loss, velocity_pred, perplexity, quantized, encodings


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def weighted_sequence_loss(predictions, targets, threshold=1.5, weight_high=3.0):
    """MSE pesata per picchi di velocità"""
    errors = (predictions - targets) ** 2
    is_peak = torch.abs(targets) > threshold
    weights = torch.where(is_peak,
                         torch.tensor(weight_high, device=targets.device),
                         torch.tensor(1.0, device=targets.device))
    return torch.mean(weights * errors)


def train_epoch(model, dataloader, optimizer, device, epoch, config):
    """Training epoch end-to-end"""
    model.train()
    
    # VQ Weight Scheduling
    warmup_epochs = config.get('warmup_epochs', 30)
    max_epochs = config.get('max_epochs', 200)
    
    if epoch < warmup_epochs:
        vq_weight = 0.001 * (epoch / warmup_epochs)
        phase = "WARMUP"
    else:
        progress = (epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
        vq_weight = 0.001 + 0.1 * progress  # Max 0.101
        phase = "VQ_ACTIVE"
    
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{phase}]")
    
    for neural_data, velocity_seq in pbar:
        neural_data = neural_data.to(device)  # (B, num_neurons, 60)
        velocity_seq = velocity_seq.to(device)  # (B, 60)
        
        # Forward pass (process entire batch)
        # neural_data: (B, num_neurons, 60) - encoder expects this format
        vq_loss, velocity_pred, perplexity, _, _ = model(neural_data)
        # velocity_pred: (B, 60)
        
        # Reconstruction loss (velocity prediction)
        if config.get('use_weighted_loss', False):
            recon_loss = weighted_sequence_loss(
                velocity_pred, velocity_seq,
                threshold=config.get('peak_threshold', 1.5),
                weight_high=config.get('peak_weight', 5.0)
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
        
        # Accumulate
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
        
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
    """Evaluation"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_vq_loss = 0
    total_perplexity = 0
    num_samples = 0
    
    with torch.no_grad():
        for neural_data, velocity_seq in tqdm(dataloader, desc="Evaluating"):
            neural_data = neural_data.to(device)  # (B, num_neurons, 60)
            velocity_seq = velocity_seq.to(device)  # (B, 60)
            
            # Forward (process entire batch)
            vq_loss, velocity_pred, perplexity, _, _ = model(neural_data)
            # velocity_pred: (B, 60)
            
            # Ensure predictions are 2D (batch, 60)
            pred_np = velocity_pred.cpu().numpy()
            target_np = velocity_seq.cpu().numpy()
            
            if pred_np.ndim == 1:
                pred_np = pred_np.reshape(1, -1)
            if target_np.ndim == 1:
                target_np = target_np.reshape(1, -1)
            
            all_predictions.append(pred_np)
            all_targets.append(target_np)
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            num_samples += velocity_seq.shape[0]
    
    # Concatenate instead of array to handle variable batch sizes
    predictions = np.concatenate(all_predictions, axis=0)  # (N, 60)
    targets = np.concatenate(all_targets, axis=0)  # (N, 60)
    
    # Metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    # Global correlation
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    global_corr = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # Per-sample correlation
    correlations_per_sample = []
    for i in range(len(predictions)):
        if np.std(predictions[i]) > 1e-8 and np.std(targets[i]) > 1e-8:
            corr, _ = pearsonr(predictions[i], targets[i])
            if not np.isnan(corr):
                correlations_per_sample.append(corr)
    
    avg_corr_per_sample = np.mean(correlations_per_sample) if correlations_per_sample else 0
    
    # Per-timestep correlation
    correlations_per_timestep = []
    for t in range(60):
        if np.std(predictions[:, t]) > 1e-8 and np.std(targets[:, t]) > 1e-8:
            corr = np.corrcoef(predictions[:, t], targets[:, t])[0, 1]
            if not np.isnan(corr):
                correlations_per_timestep.append(corr)
    
    avg_corr_per_timestep = np.mean(correlations_per_timestep) if correlations_per_timestep else 0
    
    # Peak metrics
    peak_mask = np.abs(targets) > 1.5
    if np.sum(peak_mask) > 0:
        peak_mse = np.mean((predictions[peak_mask] - targets[peak_mask]) ** 2)
    else:
        peak_mse = np.nan
    
    return {
        'test/mse': mse,
        'test/mae': mae,
        'test/global_correlation': global_corr,
        'test/avg_corr_per_sample': avg_corr_per_sample,
        'test/avg_corr_per_timestep': avg_corr_per_timestep,
        'test/peak_mse': peak_mse if not np.isnan(peak_mse) else 0,
        'test/vq_loss': total_vq_loss / num_samples if num_samples > 0 else 0,
        'test/perplexity': total_perplexity / num_samples if num_samples > 0 else 0,
    }, predictions, targets, correlations_per_timestep


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_sequence_examples(predictions, targets, epoch, n_samples=6):
    """Plot velocity prediction examples"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    amplitudes = [np.max(targets[i]) - np.min(targets[i]) for i in range(len(targets))]
    top_indices = np.argsort(amplitudes)[-n_samples:][::-1]
    
    for plot_idx, idx in enumerate(top_indices):
        ax = axes[plot_idx]
        
        pred_seq = predictions[idx]
        target_seq = targets[idx]
        timesteps = np.arange(60)
        
        ax.plot(timesteps, target_seq, 'b-', linewidth=2, label='Target', alpha=0.8)
        ax.plot(timesteps, pred_seq, 'r--', linewidth=2, label='Prediction', alpha=0.8)
        ax.fill_between(timesteps, target_seq, pred_seq, alpha=0.2, color='gray')
        
        corr = np.corrcoef(pred_seq, target_seq)[0, 1]
        mse = np.mean((pred_seq - target_seq) ** 2)
        amplitude = amplitudes[idx]
        
        ax.set_title(f'Sample {idx}: r={corr:.3f}, MSE={mse:.4f}, Amp={amplitude:.2f}',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=9)
        ax.set_ylabel('Velocity (normalized)', fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Epoch {epoch}: Behavioral Velocity Prediction (End-to-End VQ-VAE)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = {
        # Data
        'session_ids': [TRAINING_SESSION_IDS[0]],  # Cambia per più sessioni
        'window_size': 60,
        'stride': 30,
        'test_split': 0.2,
        'normalize': True,
        
        # Training
        'batch_size': 16,
        'num_epochs': 150,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        
        # VQ scheduling
        'warmup_epochs': 30,
        'max_epochs': 150,
        
        # Loss
        'use_weighted_loss': False,
        'peak_threshold': 1.5,
        'peak_weight': 3.0,
        
        # Paths
        'save_dir': './results/behavioral_vqvae_end2end',
        
        # System
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 0,
        'seed': 42,
        
        **MODEL_CONFIG
    }
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device(config['device'])
    print(f"🖥️  Device: {device}")
    
    # Create save dir
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*70)
    print("📊 LOADING DATA")
    print("="*70)
    
    train_loader, test_loader, dataset_info = create_sequence_dataloaders(
        session_ids=config['session_ids'],
        batch_size=config['batch_size'],
        test_split=config['test_split'],
        window_size=config['window_size'],
        stride=config['stride'],
        num_workers=config['num_workers'],
        normalize=config['normalize']
    )
    
    print(f"\n✅ Data loaded:")
    print(f"   Sessions: {len(config['session_ids'])} session(s) - {config['session_ids']}")
    print(f"   Train: {dataset_info['train_samples']}")
    print(f"   Test: {dataset_info['test_samples']}")
    print(f"   Neural shape: {dataset_info['neural_shape']}")
    print(f"   Velocity shape: {dataset_info['velocity_shape']}")
    
    num_neurons = dataset_info['neural_shape'][0]
    print(f"   Number of neurons: {num_neurons}")
    
    # Update config
    MODEL_CONFIG['num_neurons'] = num_neurons
    config['num_neurons'] = num_neurons
    
    # Create model
    print("\n" + "="*70)
    print("🧠 CREATING END-TO-END BEHAVIORAL VQ-VAE")
    print("="*70)
    
    model = BehavioralVQVAEEnd2End(
        num_neurons=num_neurons,
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ Model created:")
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   All components: TRAINABLE (end-to-end)")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )
    
    # Wandb
    session_info = f"{len(config['session_ids'])}sess" if len(config['session_ids']) > 1 else f"sess{config['session_ids'][0]}"
    wandb.init(
        project="calcium-behavioral-vqvae-end2end",
        name=f"end2end_{MODEL_CONFIG['num_embeddings']}_{session_info}",
        config=config,
        tags=['behavioral', 'end2end', 'velocity', 'all_trainable']
    )
    
    # Training loop
    print("\n" + "="*70)
    print("🚀 STARTING END-TO-END TRAINING")
    print("="*70)
    
    best_correlation = -np.inf
    best_epoch = 0
    patience = 50
    patience_counter = 0
    best_test_mse = np.inf
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, config)
        
        # Evaluate
        if epoch % 5 == 0:
            test_metrics, predictions, targets, corrs_per_timestep = evaluate(
                model, test_loader, device
            )
            
            # Update learning rate scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            all_metrics = {
                **train_metrics,
                **test_metrics,
                'epoch': epoch,
                'learning_rate': current_lr
            }
            wandb.log(all_metrics)
            
            print(f"\n📊 Results:")
            print(f"   Train Loss: {train_metrics['train/recon_loss']:.6f}")
            print(f"   Test MSE: {test_metrics['test/mse']:.6f}")
            print(f"   Global Corr: {test_metrics['test/global_correlation']:.4f}")
            print(f"   Avg Corr/Sample: {test_metrics['test/avg_corr_per_sample']:.4f}")
            print(f"   Peak MSE: {test_metrics['test/peak_mse']:.6f}")
            
            # Visualize
            if epoch % 20 == 0:
                fig = plot_sequence_examples(predictions, targets, epoch, n_samples=6)
                wandb.log({'sequence_examples': wandb.Image(fig), 'epoch': epoch})
                plt.close(fig)
            
            # Save best (based on correlation)
            current_corr = test_metrics['test/global_correlation']
            current_mse = test_metrics['test/mse']
            
            if current_corr > best_correlation:
                best_correlation = current_corr
                best_test_mse = current_mse
                best_epoch = epoch
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_config': MODEL_CONFIG,
                    'config': config,
                    'best_correlation': best_correlation,
                    'best_test_mse': best_test_mse,
                    'test_metrics': test_metrics
                }
                
                checkpoint_path = save_dir / f'best_behavioral_end2end_{MODEL_CONFIG["num_embeddings"]}_{session_info}.pth'
                torch.save(checkpoint, checkpoint_path)
                print(f"\n💾 Saved best model (corr={best_correlation:.4f}, MSE={best_test_mse:.4f}, epoch={epoch})")
            else:
                patience_counter += 1
                print(f"   ⚠️  No improvement ({patience_counter}/{patience} patience)")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                print(f"   Best model was at epoch {best_epoch}")
                print(f"   Best correlation: {best_correlation:.4f}")
                print(f"   Best test MSE: {best_test_mse:.4f}")
                break
        else:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({**train_metrics, 'epoch': epoch, 'learning_rate': current_lr})
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE")
    print("="*70)
    print(f"Best correlation: {best_correlation:.4f}")
    
    wandb.finish()


if __name__ == "__main__":
    main()

