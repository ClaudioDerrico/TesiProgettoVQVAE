#!/usr/bin/env python3
"""
Behavioral Sequence Prediction - APPROCCIO IBRIDO OTTIMIZZATO

INSIGHT CHIAVE:
- Il codebook È GIÀ allenato su tutte le sessioni → contiene informazione
- Il problema è come AGGREGARE e DECODIFICARE le features quantizzate
- Soluzione: Decoder MOLTO più potente con attention multi-head

Migliorie rispetto alla versione precedente:
1. ✅ Multi-Head Attention invece di aggregazione naive
2. ✅ Transformer Decoder invece di CNN semplice  
3. ✅ Positional encoding per catturare ordine temporale
4. ✅ Residual connections profonde
5. ✅ Loss function ancora più sofisticata
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
import math

from models.vqvae import CalciumVQVAE
from datasets.sequence_behavioral import create_sequence_dataloaders
from datasets.calcium import TRAINING_SESSION_IDS


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


# ============================================================================
# IMPROVED AGGREGATION WITH MULTI-HEAD ATTENTION
# ============================================================================

class NeuronAggregationModule(nn.Module):
    """
    Aggrega features di neuroni multipli usando Multi-Head Self-Attention
    
    Invece di semplice concatenazione/pooling, usa attention per pesare
    quali neuroni sono più importanti per la predizione
    """
    
    def __init__(self, embedding_dim, num_heads=8, dropout=0.1):
        super(NeuronAggregationModule, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention per aggregare neuroni
        self.neuron_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # LayerNorm + Feedforward
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_neurons, seq_len, embedding_dim)
               Features quantizzate da tutti i neuroni
        
        Returns:
            aggregated: (batch, seq_len, embedding_dim)
        """
        batch_size, num_neurons, seq_len, embedding_dim = x.shape
        
        # Reshape: (batch * seq_len, num_neurons, embedding_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, num_neurons, embedding_dim)
        x = x.view(batch_size * seq_len, num_neurons, embedding_dim)
        
        # ⚠️ PyTorch <1.9: MultiheadAttention expects (seq_len, batch, embed_dim)
        # Permute to (num_neurons, batch*seq_len, embedding_dim)
        x = x.permute(1, 0, 2)  # (num_neurons, batch*seq_len, embedding_dim)
        
        # Self-attention across neurons
        attn_out, _ = self.neuron_attention(x, x, x)
        
        # Permute back: (batch*seq_len, num_neurons, embedding_dim)
        x = attn_out.permute(1, 0, 2)  # (batch*seq_len, num_neurons, embedding_dim)
        attn_out = x
        
        x = self.norm1(x + attn_out)
        
        # Feedforward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Average pool across neurons
        x = x.mean(dim=1)  # (batch * seq_len, embedding_dim)
        
        # Reshape back
        x = x.view(batch_size, seq_len, embedding_dim)
        
        return x


# ============================================================================
# TRANSFORMER DECODER FOR VELOCITY SEQUENCE
# ============================================================================

class TransformerVelocityDecoder(nn.Module):
    """
    Transformer-based decoder per sequenze velocità
    
    Architecture:
    1. Positional encoding per features temporali
    2. Multi-layer Transformer decoder
    3. Output projection a velocità
    """
    
    def __init__(self, embedding_dim, num_layers=4, num_heads=8, 
                 hidden_dim=512, dropout=0.2, output_length=60):
        super(TransformerVelocityDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.output_length = output_length
        
        print(f"\n🔨 Building Transformer Velocity Decoder:")
        print(f"   Input: (batch, 15, {embedding_dim})")
        print(f"   Num layers: {num_layers}")
        print(f"   Num heads: {num_heads}")
        print(f"   Hidden dim: {hidden_dim}")
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=200)
        
        # Input projection (se necessario aumentare dimensionalità)
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Learnable query embeddings per output sequence
        self.query_embed = nn.Parameter(torch.randn(1, output_length, hidden_dim))
        self.query_pos = PositionalEncoding(hidden_dim, max_len=output_length)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._initialize_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total decoder params: {total_params:,}")
    
    def _initialize_weights(self):
        """Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, memory):
        """
        Args:
            memory: (batch, 15, embedding_dim) - encoded temporal features
        
        Returns:
            velocity_seq: (batch, 60)
        """
        batch_size = memory.shape[0]
        
        # Project input
        memory = self.input_proj(memory)  # (batch, 15, hidden_dim)
        
        # Add positional encoding to memory
        memory = self.pos_encoder(memory)  # (batch, 15, hidden_dim)
        
        # Prepare queries
        queries = self.query_embed.expand(batch_size, -1, -1)  # (batch, 60, hidden_dim)
        queries = self.query_pos(queries)
        
        # ⚠️ PyTorch <1.9: Transformer expects (seq_len, batch, features)
        # Permute: (seq_len, batch, features)
        memory = memory.permute(1, 0, 2)  # (15, batch, hidden_dim)
        queries = queries.permute(1, 0, 2)  # (60, batch, hidden_dim)
        
        # Transformer decoding
        output = self.transformer_decoder(queries, memory)  # (60, batch, hidden_dim)
        
        # Permute back to (batch, 60, hidden_dim)
        output = output.permute(1, 0, 2)
        
        # Project to velocity
        velocity_seq = self.output_proj(output)  # (batch, 60, 1)
        velocity_seq = velocity_seq.squeeze(-1)  # (batch, 60)
        
        return velocity_seq


# ============================================================================
# MAIN MODEL WITH IMPROVED ARCHITECTURE
# ============================================================================

class ImprovedBehavioralVQVAE(nn.Module):
    """
    VQ-VAE migliorato per predizione comportamentale
    
    Improvements:
    - Multi-head attention per aggregazione neuroni
    - Transformer decoder invece di CNN
    - Positional encoding
    """
    
    def __init__(self, pretrained_model, freeze_encoder=True, freeze_codebook=True,
                 num_attention_heads=8, num_transformer_layers=4, 
                 hidden_dim=512, dropout=0.2):
        super(ImprovedBehavioralVQVAE, self).__init__()
        
        # Frozen parts
        self.encoder = pretrained_model.encoder
        self.pre_quantization_conv = pretrained_model.pre_quantization_conv
        self.vector_quantization = pretrained_model.vector_quantization
        
        if freeze_encoder:
            print("🔒 Freezing encoder...")
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.pre_quantization_conv.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.pre_quantization_conv.eval()
        
        if freeze_codebook:
            print("🔒 Freezing codebook...")
            for param in self.vector_quantization.parameters():
                param.requires_grad = False
            self.vector_quantization.eval()
        
        # Get embedding dimension
        self.embedding_dim = pretrained_model.vector_quantization.e_dim
        
        # NEW: Neuron Aggregation Module
        self.neuron_aggregator = None  # Built at first forward
        
        # NEW: Transformer Decoder
        self.decoder = None  # Built at first forward
        
        self.num_attention_heads = num_attention_heads
        self.num_transformer_layers = num_transformer_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        print(f"\n📊 ImprovedBehavioralVQVAE initialized:")
        print(f"   Embedding dim: {self.embedding_dim}")
        print(f"   Attention heads: {num_attention_heads}")
        print(f"   Transformer layers: {num_transformer_layers}")
    
    def _build_decoder(self, num_neurons, device):
        """Build aggregator + decoder"""
        print(f"\n🔨 Building decoder for {num_neurons} neurons...")
        
        # Neuron aggregation module
        self.neuron_aggregator = NeuronAggregationModule(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_attention_heads,
            dropout=self.dropout
        ).to(device)
        
        # Transformer decoder
        self.decoder = TransformerVelocityDecoder(
            embedding_dim=self.embedding_dim,
            num_layers=self.num_transformer_layers,
            num_heads=self.num_attention_heads,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            output_length=60
        ).to(device)
        
        trainable_agg = sum(p.numel() for p in self.neuron_aggregator.parameters())
        trainable_dec = sum(p.numel() for p in self.decoder.parameters())
        print(f"✅ Decoder built:")
        print(f"   Aggregator: {trainable_agg:,} params")
        print(f"   Decoder: {trainable_dec:,} params")
        print(f"   Total trainable: {trainable_agg + trainable_dec:,} params")
    
    def forward(self, x):
        """
        Args:
            x: (num_neurons, 1, 60) or (batch, num_neurons, 60)
        
        Returns:
            velocity_seq: (batch, 60) or (60,)
        """
        # Handle different input shapes
        if x.dim() == 3 and x.shape[1] == 1:
            # Single sample: (num_neurons, 1, 60)
            num_neurons = x.shape[0]
            x = x  # Keep as is
            squeeze_output = True
            batch_mode = False
        elif x.dim() == 3:
            # Batch: (batch, num_neurons, 60)
            batch_size, num_neurons, time_len = x.shape
            # Reshape to process each sample
            x = x.view(batch_size * num_neurons, 1, time_len)
            squeeze_output = False
            batch_mode = True
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        device = x.device
        
        # Build decoder if needed
        if self.decoder is None:
            self._build_decoder(num_neurons, device)
        
        # ====================================================================
        # ENCODE + QUANTIZE (Frozen)
        # ====================================================================
        
        self.encoder.eval()
        self.pre_quantization_conv.eval()
        self.vector_quantization.eval()
        
        with torch.set_grad_enabled(self.training):
            z = self.encoder(x)  # (num_neurons or batch*num_neurons, hidden, 15)
            z = self.pre_quantization_conv(z)
            z = F.layer_norm(z, [z.size(1), z.size(2)])
            z = torch.clamp(z, min=-10.0, max=10.0)
            
            _, quantized, _, _, _ = self.vector_quantization(z)
            # quantized: (num_neurons or batch*num_neurons, embedding_dim, 15)
        
        # ====================================================================
        # RESHAPE FOR AGGREGATION
        # ====================================================================
        
        if batch_mode:
            # Reshape back to batch
            quantized = quantized.view(batch_size, num_neurons, self.embedding_dim, 15)
            # Permute to (batch, num_neurons, seq_len, embedding_dim)
            quantized = quantized.permute(0, 1, 3, 2)  # (batch, num_neurons, 15, embedding_dim)
        else:
            # Single sample
            # (num_neurons, embedding_dim, 15) → (1, num_neurons, 15, embedding_dim)
            quantized = quantized.permute(0, 2, 1).unsqueeze(0)
        
        # ====================================================================
        # AGGREGATE NEURONS with Attention
        # ====================================================================
        
        aggregated = self.neuron_aggregator(quantized)  # (batch, 15, embedding_dim)
        
        # ====================================================================
        # DECODE with Transformer
        # ====================================================================
        
        velocity_seq = self.decoder(aggregated)  # (batch, 60)
        
        if squeeze_output:
            velocity_seq = velocity_seq.squeeze(0)  # (60,)
        
        return velocity_seq
    
    def get_frozen_params_count(self):
        """Count parameters"""
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return frozen, trainable


# ============================================================================
# ADVANCED LOSS FUNCTION
# ============================================================================

def advanced_sequence_loss(predictions, targets, alpha=0.25, beta=0.25, gamma=0.25, delta=0.25):
    """
    Loss ultra-sofisticata per sequenze velocità
    
    Components:
    1. MSE: Accuratezza punto-a-punto
    2. Correlation: Forma del segnale
    3. Spectral: Similarità nello spazio delle frequenze
    4. Dynamic Time Warping inspired: Flessibilità temporale
    """
    # 1. MSE
    mse_loss = F.mse_loss(predictions, targets)
    
    # 2. Correlation
    pred_centered = predictions - predictions.mean(dim=1, keepdim=True)
    target_centered = targets - targets.mean(dim=1, keepdim=True)
    
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = (
        torch.sqrt((pred_centered ** 2).sum(dim=1)) * 
        torch.sqrt((target_centered ** 2).sum(dim=1)) + 1e-8
    )
    correlation = numerator / denominator
    corr_loss = -correlation.mean()
    
    # 3. Spectral loss (similarità nello spazio delle frequenze)
    pred_fft = torch.fft.rfft(predictions, dim=1)
    target_fft = torch.fft.rfft(targets, dim=1)
    
    # Magnitude spectrum
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    spectral_loss = F.mse_loss(pred_mag, target_mag)
    
    # 4. Derivative loss (smoothness)
    pred_diff = predictions[:, 1:] - predictions[:, :-1]
    target_diff = targets[:, 1:] - targets[:, :-1]
    derivative_loss = F.mse_loss(pred_diff, target_diff)
    
    # Combine
    total_loss = alpha * mse_loss + beta * corr_loss + gamma * spectral_loss + delta * derivative_loss
    
    return total_loss, {
        'mse': mse_loss.item(),
        'corr_loss': corr_loss.item(),
        'actual_corr': correlation.mean().item(),
        'spectral': spectral_loss.item(),
        'derivative': derivative_loss.item()
    }


# ============================================================================
# LOADING PRETRAINED
# ============================================================================

def load_pretrained_vqvae(checkpoint_path, device):
    """Load pretrained VQ-VAE"""
    print(f"\n🔄 Loading pretrained model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']
    
    print(f"   Pretrained config:")
    for key, value in model_config.items():
        print(f"     {key}: {value}")
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded pretrained model (epoch {checkpoint.get('epoch', 'N/A')})")
    
    return model, model_config


# ============================================================================
# TRAINING (same as before but with new model)
# ============================================================================

def train_epoch(model, train_loader, optimizer, device, epoch, config):
    """Training epoch"""
    model.train()
    model.neuron_aggregator.train()
    model.decoder.train()
    
    total_loss = 0
    total_corr = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for neural_data, velocity_seq in pbar:
        neural_data = neural_data.to(device)
        velocity_seq = velocity_seq.to(device)
        
        # Forward
        velocity_pred = model(neural_data)
        
        # Loss
        loss, loss_components = advanced_sequence_loss(
            velocity_pred, velocity_seq,
            alpha=config['loss_alpha'],
            beta=config['loss_beta'],
            gamma=config['loss_gamma'],
            delta=config['loss_delta']
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_corr += loss_components['actual_corr']
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'corr': f'{loss_components["actual_corr"]:.3f}'
        })
    
    return {
        'train/loss': total_loss / num_batches,
        'train/corr': total_corr / num_batches
    }


def evaluate(model, test_loader, device, config):
    """Evaluation"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for neural_data, velocity_seq in tqdm(test_loader, desc="Evaluating"):
            neural_data = neural_data.to(device)
            velocity_pred = model(neural_data)
            
            all_predictions.append(velocity_pred.cpu().numpy())
            all_targets.append(velocity_seq.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    global_corr = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    correlations = []
    for i in range(len(predictions)):
        if np.std(predictions[i]) > 1e-8 and np.std(targets[i]) > 1e-8:
            corr, _ = pearsonr(predictions[i], targets[i])
            if not np.isnan(corr):
                correlations.append(corr)
    
    avg_corr = np.mean(correlations) if correlations else 0
    
    return {
        'test/mse': mse,
        'test/mae': mae,
        'test/global_correlation': global_corr,
        'test/avg_correlation': avg_corr,
    }, predictions, targets


# ... (plot functions same as before) ...

def main():
    config = {
        'pretrained_checkpoint': './results/best_single_neuron_model_2048_pt2.pth',
        'session_id': TRAINING_SESSION_IDS[0],
        'batch_size': 16,
        'num_epochs': 150,
        'learning_rate': 3e-4,
        'weight_decay': 1e-5,
        
        # Model
        'num_attention_heads': 8,
        'num_transformer_layers': 4,
        'hidden_dim': 512,
        'dropout': 0.2,
        
        # Loss
        'loss_alpha': 0.25,
        'loss_beta': 0.25,
        'loss_gamma': 0.25,
        'loss_delta': 0.25,
        
        'save_dir': './results/transformer_behavioral_vqvae',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
    }
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device(config['device'])
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("🚀 TRANSFORMER BEHAVIORAL VQ-VAE (IMPROVED)")
    print("="*70)
    
    # Load pretrained
    pretrained_model, pretrained_config = load_pretrained_vqvae(
        config['pretrained_checkpoint'], device
    )
    
    # Load data
    train_loader, test_loader, dataset_info = create_sequence_dataloaders(
        session_ids=[config['session_id']],
        batch_size=config['batch_size'],
        test_split=0.2,
        window_size=60,
        stride=30,
        num_workers=0,
        normalize=True
    )
    
    num_neurons = dataset_info['neural_shape'][0]
    print(f"\n📊 Data: {num_neurons} neurons, {dataset_info['train_samples']} train samples")
    
    # Create model
    model = ImprovedBehavioralVQVAE(
        pretrained_model=pretrained_model,
        num_attention_heads=config['num_attention_heads'],
        num_transformer_layers=config['num_transformer_layers'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(device)
    
    # Build decoder
    sample_batch = next(iter(train_loader))[0]  # Get neural data
    model._build_decoder(num_neurons, device)
    
    frozen, trainable = model.get_frozen_params_count()
    print(f"\n📊 Parameters: {frozen:,} frozen, {trainable:,} trainable")
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=1e-6
    )
    
    # WandB
    wandb.init(
        project="calcium-transformer-behavioral",
        name=f"transformer_session_{config['session_id']}",
        config=config
    )
    
    # Training
    best_corr = -np.inf
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, config)
        
        if epoch % 5 == 0:
            test_metrics, preds, targets = evaluate(model, test_loader, device, config)
            
            wandb.log({**train_metrics, **test_metrics, 'epoch': epoch})
            
            print(f"   Train Corr: {train_metrics['train/corr']:.4f}")
            print(f"   Test Corr: {test_metrics['test/global_correlation']:.4f}")
            print(f"   Test MSE: {test_metrics['test/mse']:.4f}")
            
            current_corr = test_metrics['test/global_correlation']
            if current_corr > best_corr:
                best_corr = current_corr
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'best_correlation': best_corr
                }, save_dir / 'best_transformer_model.pth')
                
                print(f"   💾 Saved (corr={best_corr:.4f})")
        
        scheduler.step()
    
    print(f"\n🎉 Best correlation: {best_corr:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()