#!/usr/bin/env python3
"""
Behavioral Sequence Prediction con VQ-VAE Pretrained

Usa encoder + codebook FROZEN da modello pretrained
Addestra solo il decoder per predire sequenze di velocità (60 timesteps)

Architecture flow:
1. Input: (num_neurons, 1, 60) - tutti i neuroni della sessione
2. Encoder (frozen): (num_neurons, embedding_dim, 15)
3. Codebook (frozen): (num_neurons, embedding_dim, 15) quantizzato
4. Aggregazione: (1, num_neurons*embedding_dim, 15)
5. Decoder NEW: (1, 60) - sequenza velocità completa!
"""

import sys
import os
from pathlib import Path
import random

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
# BEHAVIORAL SEQUENCE DECODER
# ============================================================================

class BehavioralSequenceDecoder(nn.Module):
    """
    Decoder CONV-BASED per sequenze comportamentali
    
    Strategy: Conv1D reduction + temporal upsampling
    NO global pooling - mantiene info spaziale!
    """
    
    def __init__(self, input_channels, hidden_dim=256, dropout_rate=0.3):
        super(BehavioralSequenceDecoder, self).__init__()
        
        self.input_channels = input_channels
        
        print(f"\n🔨 Building CONV-BASED Behavioral Sequence Decoder:")
        print(f"   Input channels: {input_channels}")
        print(f"   Strategy: Conv reduction + Temporal upsample (NO global pooling)")
        
        # =====================================================================
        # Stage 1: Dimensionality reduction with Conv1D
        # =====================================================================
        # Reduce channels gradually: 23168 → 512 → 256
        
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
        
        # =====================================================================
        # Stage 2: Temporal upsampling 15 → 30 → 60
        # =====================================================================
        
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
        
        # =====================================================================
        # Stage 3: Refinement
        # =====================================================================
        
        self.refinement = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # =====================================================================
        # Stage 4: Output
        # =====================================================================
        
        self.output = nn.Conv1d(32, 1, kernel_size=1)
        
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # ⭐⭐
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Architecture: {input_channels} → 512 → 256 → [upsample] → 64 → 32 → 1")
        print(f"   Total decoder params: {total_params:,}")
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_neurons*embedding_dim, 15)
        
        Returns:
            velocity_seq: (batch, 60)
        """
        # Reduce dimensions
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


class BehavioralSequenceVQVAE(nn.Module):
    """
    VQ-VAE per predizione sequenze comportamentali
    
    - Encoder + Codebook: FROZEN (pretrained)
    - Decoder: NEW, trainable
    """
    
    def __init__(self, pretrained_model, freeze_encoder=True, freeze_codebook=True,
                 hidden_dim=256, dropout_rate=0.3):
        super(BehavioralSequenceVQVAE, self).__init__()
        
        # ✅ FROZEN: Encoder + Pre-quantization
        self.encoder = pretrained_model.encoder
        self.pre_quantization_conv = pretrained_model.pre_quantization_conv
        
        if freeze_encoder:
            print("🔒 Freezing encoder...")
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.pre_quantization_conv.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.pre_quantization_conv.eval()
        
        # ✅ FROZEN: Codebook
        self.vector_quantization = pretrained_model.vector_quantization
        
        if freeze_codebook:
            print("🔒 Freezing codebook...")
            for param in self.vector_quantization.parameters():
                param.requires_grad = False
            self.vector_quantization.eval()
        
        # ✅ NEW: Behavioral Sequence Decoder
        self.embedding_dim = pretrained_model.vector_quantization.e_dim
        self.decoder = None  # Built at first forward
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        print(f"\n📊 BehavioralSequenceVQVAE initialized:")
        print(f"   Embedding dim: {self.embedding_dim}")
        print(f"   Decoder: will be built at first forward")
    
    def _build_decoder(self, num_neurons, device):
        """Build decoder basato sul numero di neuroni della sessione"""
        input_channels = num_neurons * self.embedding_dim
        
        self.decoder = BehavioralSequenceDecoder(
            input_channels=input_channels,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate
        ).to(device)
        
        trainable = sum(p.numel() for p in self.decoder.parameters())
        print(f"✅ Decoder built: {trainable:,} trainable params")
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: (num_neurons, 1, 60) neural data
        
        Returns:
            velocity_seq: (1, 60) se batch, o (60,) se single sample
            features: (optional) quantized features
        """
        num_neurons = x.shape[0]
        device = x.device
        
        # Build decoder if needed
        if self.decoder is None:
            self._build_decoder(num_neurons, device)
        
        # ========================================================================
        # ENCODE + QUANTIZE (Frozen)
        # ========================================================================
        
        # Set to eval mode for frozen parts
        self.encoder.eval()
        self.pre_quantization_conv.eval()
        self.vector_quantization.eval()
        
        with torch.set_grad_enabled(self.training):  # Allow gradients to flow through
            z = self.encoder(x)  # (num_neurons, hidden_dim, 15)
            z = self.pre_quantization_conv(z)  # (num_neurons, embedding_dim, 15)
            
            # Normalize
            z = F.layer_norm(z, [z.size(1), z.size(2)])
            z = torch.clamp(z, min=-10.0, max=10.0)
            
            # Quantize
            vq_loss, quantized, perplexity, encodings, encoding_indices = \
                self.vector_quantization(z)
            # quantized: (num_neurons, embedding_dim, 15)
        
        # ========================================================================
        # AGGREGATE neurons
        # ========================================================================
        
        num_neurons_actual, embedding_dim, compressed_time = quantized.shape
        
        # Reshape: (num_neurons, embedding_dim, 15) → (1, num_neurons*embedding_dim, 15)
        quantized_aggregated = quantized.reshape(1, num_neurons_actual * embedding_dim, compressed_time)
        
        # ========================================================================
        # DECODE to velocity sequence (Trainable)
        # ========================================================================
        
        velocity_seq = self.decoder(quantized_aggregated)  # (1, 60)
        
        if return_features:
            return velocity_seq, quantized_aggregated, vq_loss, perplexity
        else:
            return velocity_seq
    
    def get_frozen_params_count(self):
        """Count frozen vs trainable parameters"""
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return frozen, trainable


# ============================================================================
# LOADING PRETRAINED MODEL
# ============================================================================

def load_pretrained_vqvae(checkpoint_path, device):
    """Load pretrained VQ-VAE model"""
    print(f"\n🔄 Loading pretrained model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    model_config = checkpoint['model_config']
    
    print(f"   Pretrained config:")
    for key, value in model_config.items():
        print(f"     {key}: {value}")
    
    # Create model
    model = CalciumVQVAE(
        num_neurons=model_config['num_neurons'],
        num_hiddens=model_config['num_hiddens'],
        num_residual_layers=model_config['num_residual_layers'],
        num_residual_hiddens=model_config['num_residual_hiddens'],
        num_embeddings=model_config['num_embeddings'],
        embedding_dim=model_config['embedding_dim'],
        commitment_cost=model_config['commitment_cost'],
        dropout_rate=model_config.get('dropout_rate', 0.0),
        use_quantizer=model_config.get('use_quantizer', True)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded pretrained model (epoch {checkpoint.get('epoch', 'N/A')})")
    
    if 'best_correlation' in checkpoint:
        print(f"   Best correlation: {checkpoint['best_correlation']:.4f}")
    
    return model, model_config


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def weighted_sequence_loss(predictions, targets, threshold=1.5, weight_high=3.0):
    """MSE pesata che dà più importanza ai timesteps con picchi"""
    
    # Errori
    errors = (predictions - targets) ** 2
    
    # Identifica picchi
    is_peak = torch.abs(targets) > threshold
    
    # Pesi
    weights = torch.where(is_peak,
                         torch.tensor(weight_high, device=targets.device),
                         torch.tensor(1.0, device=targets.device))
    
    # Loss pesata
    weighted_loss = torch.mean(weights * errors)
    
    return weighted_loss


def precompute_features(model, dataloader, device):
    """
    Precompute quantized features from frozen encoder+quantizer
    This is done ONCE at the beginning to massively speed up training!
    
    Returns:
        cached_features: list of (quantized_aggregated, velocity_seq) tuples
    """
    print("\n🔥 PRECOMPUTING FEATURES (this will make training MUCH faster!)")
    print("="*70)
    
    model.eval()
    cached_features = []
    
    with torch.no_grad():
        for neural_data, velocity_seq in tqdm(dataloader, desc="Caching features"):
            batch_size = neural_data.shape[0]
            neural_data = neural_data.to(device)
            
            batch_quantized = []
            
            for i in range(batch_size):
                sample_neural = neural_data[i].unsqueeze(1)  # (num_neurons, 1, 60)
                
                # Forward through frozen encoder+quantizer
                z = model.encoder(sample_neural)
                z = model.pre_quantization_conv(z)
                z = F.layer_norm(z, [z.size(1), z.size(2)])
                z = torch.clamp(z, min=-10.0, max=10.0)
                
                _, quantized, _, _, _ = model.vector_quantization(z)
                
                # Aggregate
                num_neurons_actual, embedding_dim, compressed_time = quantized.shape
                quantized_aggregated = quantized.reshape(1, num_neurons_actual * embedding_dim, compressed_time)
                
                batch_quantized.append(quantized_aggregated)
            
            # Stack batch
            batch_quantized = torch.cat(batch_quantized, dim=0)  # (batch, num_neurons*embedding_dim, 15)
            
            # Store on CPU to save GPU memory
            cached_features.append((batch_quantized.cpu(), velocity_seq.cpu()))
    
    print(f"✅ Cached {len(cached_features)} batches of features")
    print(f"   Each batch: {cached_features[0][0].shape}")
    print("="*70)
    
    return cached_features


def train_epoch_with_cache(model, cached_features, optimizer, device, epoch, config):
    """
    Training epoch using CACHED features - MUCH FASTER!
    """
    model.decoder.train()
    
    total_loss = 0
    num_batches = len(cached_features)
    
    # Shuffle cached features for this epoch
    import random
    epoch_features = cached_features.copy()
    random.shuffle(epoch_features)
    
    pbar = tqdm(epoch_features, desc=f"Epoch {epoch}")
    
    for batch_quantized, velocity_seq in pbar:
        # Move to device
        batch_quantized = batch_quantized.to(device)
        velocity_seq = velocity_seq.to(device)
        
        # Enable gradients only for decoder
        batch_quantized.requires_grad_(True)
        
        # Forward through decoder ONLY (trainable)
        velocity_pred = model.decoder(batch_quantized)  # (batch, 60)
        
        # Loss
        if config.get('use_weighted_loss', False):
            loss = weighted_sequence_loss(
                velocity_pred, velocity_seq,
                threshold=config.get('peak_threshold', 1.5),
                weight_high=config.get('peak_weight', 3.0)
            )
        else:
            loss = F.mse_loss(velocity_pred, velocity_seq)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return {
        'train/loss': total_loss / num_batches
    }


def evaluate_with_cache(model, cached_features, device):
    """
    Evaluation using CACHED features - MUCH FASTER!
    """
    model.decoder.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_quantized, velocity_seq in tqdm(cached_features, desc="Evaluating"):
            batch_quantized = batch_quantized.to(device)
            
            # Forward through decoder only
            velocity_pred = model.decoder(batch_quantized)  # (batch, 60)
            
            all_predictions.append(velocity_pred.cpu().numpy())
            all_targets.append(velocity_seq.numpy())
    
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
    }, predictions, targets, correlations_per_timestep



def evaluate(model, dataloader, device):
    """Evaluation - OPTIMIZED"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for neural_data, velocity_seq in tqdm(dataloader, desc="Evaluating"):
            batch_size = neural_data.shape[0]
            
            neural_data = neural_data.to(device)
            velocity_seq = velocity_seq.to(device)
            
            # ✅ OPTIMIZATION: Process frozen parts without gradients
            batch_quantized = []
            
            for i in range(batch_size):
                sample_neural = neural_data[i].unsqueeze(1)
                
                # Frozen parts
                z = model.encoder(sample_neural)
                z = model.pre_quantization_conv(z)
                z = F.layer_norm(z, [z.size(1), z.size(2)])
                z = torch.clamp(z, min=-10.0, max=10.0)
                
                _, quantized, _, _, _ = model.vector_quantization(z)
                
                # Aggregate
                num_neurons_actual, embedding_dim, compressed_time = quantized.shape
                quantized_aggregated = quantized.reshape(1, num_neurons_actual * embedding_dim, compressed_time)
                
                batch_quantized.append(quantized_aggregated)
            
            # Stack and decode in batch
            batch_quantized = torch.cat(batch_quantized, dim=0)
            velocity_pred = model.decoder(batch_quantized)  # (batch, 60)
            
            all_predictions.append(velocity_pred.cpu().numpy())
            all_targets.append(velocity_seq.cpu().numpy())
    
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
    }, predictions, targets, correlations_per_timestep


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_sequence_examples(predictions, targets, epoch, n_samples=6):
    """Plot reconstruction examples"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Select high-amplitude samples
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
    
    plt.suptitle(f'Epoch {epoch}: Behavioral Sequence Prediction (Pretrained VQ-VAE)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    config = {
        # Pretrained model
        'pretrained_checkpoint': './results/best_single_neuron_model_512_pt2.pth',
        # Or use dual codebook:
        # 'pretrained_checkpoint': './results/dual_codebook/best_dual_model.pth',
        
        # Data
        'session_id': TRAINING_SESSION_IDS[0],
        'window_size': 60,
        'stride': 30,
        'test_split': 0.2,
        'normalize': True,
        
        # Training
        'batch_size': 16,  # ✅ INCREASED for faster training (was 16)
        'num_epochs': 200,
        'learning_rate': 1e-4,  # ✅ REDUCED - was too high (1e-3)
        'weight_decay': 1e-4,
        
        # Loss
        'use_weighted_loss': False,
        'peak_threshold': 1.5,
        'peak_weight': 3.0,
        
        # Decoder
        'hidden_dim': 256,
        'dropout_rate': 0.1,
        
        # Freezing
        'freeze_encoder': True,
        'freeze_codebook': True,
        
        # Paths
        'save_dir': './results/behavioral_sequence_vqvae',
        
        # System
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 0,
        'seed': 42,
    }
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device(config['device'])
    print(f"🖥️  Device: {device}")
    
    # Create save dir
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # LOAD PRETRAINED MODEL
    # ========================================================================
    
    pretrained_model, pretrained_config = load_pretrained_vqvae(
        config['pretrained_checkpoint'],
        device
    )
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
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
    
    # Get num_neurons from dataset
    num_neurons = dataset_info['neural_shape'][0]
    print(f"   Number of neurons: {num_neurons}")
    
    # ========================================================================
    # CREATE BEHAVIORAL MODEL
    # ========================================================================
    
    print("\n" + "="*70)
    print("🧠 CREATING BEHAVIORAL MODEL")
    print("="*70)
    
    model = BehavioralSequenceVQVAE(
        pretrained_model=pretrained_model,
        freeze_encoder=config['freeze_encoder'],
        freeze_codebook=config['freeze_codebook'],
        hidden_dim=config['hidden_dim'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # ✅ BUILD DECODER IMMEDIATELY with known num_neurons
    print(f"\n🔨 Building decoder for {num_neurons} neurons...")
    model._build_decoder(num_neurons, device)
    
    frozen, trainable = model.get_frozen_params_count()
    print(f"\n📊 Parameter counts:")
    print(f"   Frozen: {frozen:,}")
    print(f"   Trainable: {trainable:,} (decoder only)")
    print(f"   Total: {frozen + trainable:,}")
    
    # ========================================================================
    # OPTIMIZER
    # ========================================================================
    
    # Only optimize decoder parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )
    
    # ========================================================================
    # WANDB
    # ========================================================================
    
    wandb.init(
        project="calcium-behavioral-vqvae-sequence",
        name=f"behavioral_seq_{Path(config['pretrained_checkpoint']).stem}",
        config=config,
        tags=['behavioral', 'sequence', 'pretrained', 'frozen_encoder']
    )
    
    # ========================================================================
    # PRECOMPUTE FEATURES (ONCE!)
    # ========================================================================
    
    print("\n" + "="*70)
    print("🚀 PRECOMPUTING FEATURES FOR SPEED")
    print("="*70)
    
    # Cache features for train and test
    train_cached = precompute_features(model, train_loader, device)
    test_cached = precompute_features(model, test_loader, device)
    
    print(f"\n✅ Features cached!")
    print(f"   Train batches: {len(train_cached)}")
    print(f"   Test batches: {len(test_cached)}")
    print(f"   Now training will be MUCH faster! ⚡")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    
    best_correlation = -np.inf
    patience = 50
    patience_counter = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        
        # Train using CACHED features
        train_metrics = train_epoch_with_cache(
            model, train_cached, optimizer, device, epoch, config
        )
        
        # Update LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Evaluate
        if epoch % 5 == 0:
            test_metrics, predictions, targets, corrs_per_timestep = evaluate_with_cache(
                model, test_cached, device
            )
            
            all_metrics = {
                **train_metrics,
                **test_metrics,
                'epoch': epoch,
                'learning_rate': current_lr
            }
            wandb.log(all_metrics)
            
            print(f"\n📊 Results:")
            print(f"   Train Loss: {train_metrics['train/loss']:.6f}")
            print(f"   Test MSE: {test_metrics['test/mse']:.6f}")
            print(f"   Global Corr: {test_metrics['test/global_correlation']:.4f}")
            print(f"   Avg Corr/Sample: {test_metrics['test/avg_corr_per_sample']:.4f}")
            
            # Visualize
            if epoch % 20 == 0:
                fig = plot_sequence_examples(predictions, targets, epoch, n_samples=6)
                wandb.log({'sequence_examples': wandb.Image(fig), 'epoch': epoch})
                plt.close(fig)
            
            # Save best
            current_corr = test_metrics['test/global_correlation']
            if current_corr > best_correlation:
                best_correlation = current_corr
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'decoder_state_dict': model.decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'pretrained_config': pretrained_config,
                    'best_correlation': best_correlation,
                    'test_metrics': test_metrics
                }
                
                torch.save(checkpoint, save_dir / 'best_behavioral_model.pth')
                print(f"\n💾 Saved best model (corr={best_correlation:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        else:
            wandb.log({**train_metrics, 'epoch': epoch, 'learning_rate': current_lr})
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE")
    print("="*70)
    print(f"Best correlation: {best_correlation:.4f}")
    
    wandb.finish()


if __name__ == "__main__":
    main()