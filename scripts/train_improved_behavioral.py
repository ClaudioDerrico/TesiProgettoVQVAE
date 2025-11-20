#!/usr/bin/env python3
"""
Improved Training for Behavioral VQ-VAE with Informative Neuron Selection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from scipy.stats import pearsonr

from datasets.improved_velocity_dataset import create_informative_dataloaders
from models.vqvae import CalciumVQVAE
from datasets.calcium import TRAINING_SESSION_IDS


# ============================================================================
# IMPROVED BEHAVIORAL MODEL
# ============================================================================

class ImprovedBehavioralVQVAE(nn.Module):
    """
    Improved Behavioral VQ-VAE with better architecture
    """
    
    def __init__(self, pretrained_encoder, pretrained_vq, embedding_dim=32, 
                 freeze_encoder=True, freeze_codebook=True, use_quantization=True):
        super().__init__()
        
        self.use_quantization = use_quantization
        self.freeze_encoder = freeze_encoder
        
        self.encoder = pretrained_encoder
        self.pre_quantization_conv = nn.Conv1d(128, embedding_dim, 1, 1)
        self.vector_quantization = pretrained_vq
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.pre_quantization_conv.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        if freeze_codebook:
            for param in self.vector_quantization.parameters():
                param.requires_grad = False
            self.vector_quantization.eval()
        
        self.embedding_dim = embedding_dim
        
        # Decoder architecture
        self.decoder = nn.ModuleDict({
            'temporal_conv': nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1),
            'temporal_pool': nn.AdaptiveAvgPool1d(4),
            
            'fc1': nn.Linear(128 * 4, 256),
            'bn1': nn.BatchNorm1d(256),
            
            'fc2': nn.Linear(256, 128),
            'bn2': nn.BatchNorm1d(128),
            
            'fc3': nn.Linear(128, 64),
            'bn3': nn.BatchNorm1d(64),
            
            'fc_out': nn.Linear(64, 1),
            'skip': nn.Linear(128 * 4, 64),
            'dropout': nn.Dropout(0.2),
        })
        
        # Initialize weights
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size, num_neurons, seq_len = x.shape
        x_reshaped = x.view(batch_size * num_neurons, 1, seq_len)
        
        # Encode
        if self.freeze_encoder:
            with torch.no_grad():
                z = self.encoder(x_reshaped)
                z = self.pre_quantization_conv(z)
        else:
            z = self.encoder(x_reshaped)
            z = self.pre_quantization_conv(z)
        
        z = F.layer_norm(z, [z.size(1), z.size(2)])
        z = torch.clamp(z, min=-10.0, max=10.0)
        
        # Quantize or bypass
        if self.use_quantization:
            with torch.no_grad():
                _, quantized, _, _, _ = self.vector_quantization(z)
        else:
            # Bypass quantization - use continuous features
            quantized = z
        
        # Reshape and aggregate
        _, embed_dim, comp_time = quantized.shape
        quantized = quantized.view(batch_size, num_neurons * embed_dim, comp_time)
        
        neuron_weights = F.softmax(quantized.mean(dim=2), dim=1)
        quantized_weighted = quantized * neuron_weights.unsqueeze(2)
        quantized_agg = quantized_weighted.view(batch_size, num_neurons, embed_dim, comp_time).sum(dim=1)
        
        # Decode
        h = F.relu(self.decoder['temporal_conv'](quantized_agg))
        h = self.decoder['temporal_pool'](h)
        h_flat = h.flatten(1)
        
        skip = self.decoder['skip'](h_flat)
        
        h = self.decoder['fc1'](h_flat)
        h = self.decoder['bn1'](h)
        h = F.relu(h)
        h = self.decoder['dropout'](h)
        
        h = self.decoder['fc2'](h)
        h = self.decoder['bn2'](h)
        h = F.relu(h)
        h = self.decoder['dropout'](h)
        
        h = self.decoder['fc3'](h)
        h = self.decoder['bn3'](h)
        h = F.relu(h)
        
        h = h + skip
        h = self.decoder['dropout'](h)
        
        output = self.decoder['fc_out'](h).squeeze(-1)
        return output


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, gradient_clip=1.0):
    """Train one epoch"""
    model.train()
    model.decoder.train()
    
    # Keep encoder frozen if specified
    if model.freeze_encoder:
        model.encoder.eval()
    
    total_loss = 0
    gradient_norms = []
    
    for neural_data, velocity_target in dataloader:
        neural_data = neural_data.to(device)
        velocity_target = velocity_target.to(device)
        
        predictions = model(neural_data)
        loss = F.smooth_l1_loss(predictions, velocity_target)
        
        # L2 regularization
        l2_reg = sum(p.norm(2) for p in model.parameters() if p.requires_grad)
        loss = loss + 1e-5 * l2_reg
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient norm
        total_norm = sum(p.grad.data.norm(2).item() ** 2 
                        for p in model.parameters() if p.grad is not None) ** 0.5
        gradient_norms.append(total_norm)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
    
    return {
        'train/loss': total_loss / len(dataloader),
        'train/grad_norm': np.mean(gradient_norms),
    }


def evaluate_model(model, dataloader, device):
    """Evaluation"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for neural_data, velocity_target in dataloader:
            neural_data = neural_data.to(device)
            predictions = model(neural_data)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(velocity_target.numpy())
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    pred_std = np.std(predictions)
    
    if pred_std < 1e-6:
        correlation, r2 = 0.0, -1.0
    else:
        correlation, _ = pearsonr(predictions, targets)
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    
    return {
        'test/mse': mse,
        'test/mae': mae,
        'test/correlation': correlation,
        'test/r2': r2,
        'test/pred_std': pred_std,
        'test/target_std': np.std(targets),
    }


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_improved_model(session_id, config):
    """Main training function"""
    
    # Build experiment name
    exp_name = f"session-{session_id}"
    if not config.get('use_quantization', True):
        exp_name += "-noVQ"
    if not config.get('freeze_encoder', True):
        exp_name += "-finetune"
    
    wandb.init(
        project="calcium-velocity-improved",
        name=exp_name,
        config=config,
    )
    
    print("="*70)
    print(f"🚀 VELOCITY TRAINING - Session {session_id}")
    print(f"   Use Quantization: {config.get('use_quantization', True)}")
    print(f"   Freeze Encoder: {config.get('freeze_encoder', True)}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}\n")
    
    # Load pretrained model
    print("📦 Loading pretrained model...")
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    
    pretrained_model = CalciumVQVAE(
        num_neurons=1, num_hiddens=128, num_residual_layers=3,
        num_residual_hiddens=64, num_embeddings=32, embedding_dim=32,
        commitment_cost=2.0, dropout_rate=0.1
    )
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    pretrained_model.to(device)
    
    # Create model
    print("🧠 Creating behavioral model...")
    model = ImprovedBehavioralVQVAE(
        pretrained_encoder=pretrained_model.encoder,
        pretrained_vq=pretrained_model.vector_quantization,
        embedding_dim=32,
        freeze_encoder=config.get('freeze_encoder', True),
        freeze_codebook=config.get('freeze_codebook', True),
        use_quantization=config.get('use_quantization', True)
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"   Trainable: {trainable_params:,} | Frozen: {frozen_params:,}")
    
    # Load data
    print(f"\n📊 Loading data...")
    train_loader, test_loader, data_info = create_informative_dataloaders(
        session_id=session_id,
        top_k_neurons=config['top_k_neurons'],
        selection_method=config['selection_method'],
        batch_size=config['batch_size'],
        stride=config['stride'],
        test_split=0.2
    )
    
    # Optimizer - include encoder params if fine-tuning
    if config.get('freeze_encoder', True):
        params_to_optimize = model.decoder.parameters()
    else:
        params_to_optimize = model.parameters()
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Training loop
    print("\n🚀 Starting training...\n")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Corr':>7} | {'R²':>7} | {'LR':>10} | Status")
    print("-" * 65)
    
    best_corr = -1.0
    patience_counter = 0
    
    for epoch in range(config['max_epochs']):
        # Warmup
        if epoch < 5:
            for g in optimizer.param_groups:
                g['lr'] = config['learning_rate'] * (epoch + 1) / 5
        
        train_metrics = train_epoch(model, train_loader, optimizer, device, 
                                   config['gradient_clip'])
        test_metrics = evaluate_model(model, test_loader, device)
        scheduler.step()
        
        # Log to wandb
        metrics = {**train_metrics, **test_metrics, 'epoch': epoch, 
                  'lr': optimizer.param_groups[0]['lr']}
        wandb.log(metrics)
        
        # Status
        status = ""
        if test_metrics['test/correlation'] > best_corr:
            best_corr = test_metrics['test/correlation']
            patience_counter = 0
            status = "💾 Best"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_correlation': best_corr,
                'config': config,
            }, f'results/best_velocity_{session_id}_{exp_name}.pth')
        else:
            patience_counter += 1
        
        # Print every 5 epochs or when best
        if epoch % 5 == 0 or status:
            print(f"{epoch:5d} | {train_metrics['train/loss']:8.4f} | "
                  f"{test_metrics['test/correlation']:7.4f} | "
                  f"{test_metrics['test/r2']:7.4f} | "
                  f"{optimizer.param_groups[0]['lr']:10.2e} | {status}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break
    
    print("-" * 65)
    print(f"✅ Completed! Best correlation: {best_corr:.4f}")
    
    wandb.finish()
    return best_corr


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # ===========================================
    # EXPERIMENT CONFIGURATION
    # ===========================================
    # Esperimento 1: Con VQ (baseline)
    # Esperimento 2: Senza VQ (test diagnostico)
    # Esperimento 3: Fine-tune encoder
    # ===========================================
    
    EXPERIMENT = 2  # <-- CAMBIA QUESTO: 1, 2, o 3
    
    if EXPERIMENT == 1:
        # Baseline: VQ attivo, encoder frozen
        config = {
            'checkpoint_path': 'results/best_single_neuron_model_32.pth',
            'top_k_neurons': 50,
            'selection_method': 'correlation',
            'stride': 10,
            'freeze_encoder': True,
            'freeze_codebook': True,
            'use_quantization': True,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'max_epochs': 200,
            'patience': 30,
        }
    
    elif EXPERIMENT == 2:
        # Test: Bypass VQ per vedere se il problema è la quantizzazione
        config = {
            'checkpoint_path': 'results/best_single_neuron_model_32.pth',
            'top_k_neurons': 50,
            'selection_method': 'correlation',
            'stride': 10,
            'freeze_encoder': True,
            'freeze_codebook': True,
            'use_quantization': False,  # <-- BYPASS VQ
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'max_epochs': 200,
            'patience': 30,
        }
    
    elif EXPERIMENT == 3:
        # Fine-tune encoder (con VQ attivo)
        config = {
            'checkpoint_path': 'results/best_single_neuron_model_32.pth',
            'top_k_neurons': 50,
            'selection_method': 'correlation',
            'stride': 10,
            'freeze_encoder': False,  # <-- FINE-TUNE
            'freeze_codebook': True,
            'use_quantization': True,
            'batch_size': 32,
            'learning_rate': 1e-4,  # <-- LR più basso per fine-tuning
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'max_epochs': 200,
            'patience': 30,
        }
    
    SESSION_INDEX = 0
    session_id = TRAINING_SESSION_IDS[SESSION_INDEX]
    
    print(f"🎯 EXPERIMENT {EXPERIMENT} on session {session_id}")
    best_correlation = train_improved_model(session_id, config)
    print(f"\n📊 Final Result: {best_correlation:.4f}")