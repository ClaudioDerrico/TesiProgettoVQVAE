"""
Training Script - Sequence-to-Sequence Behavioral Prediction

Train un modello che predice l'INTERA traccia temporale di velocità (60 timesteps)
invece di un singolo valore medio.

Questo permette di catturare i picchi istantanei!
"""

import sys
import os
from pathlib import Path

# Aggiungi root al PYTHONPATH
root_dir = Path(__file__).parent.parent if Path(__file__).parent.name == 'scripts' else Path(__file__).parent
sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.sequence_behavioral_regressor import SequenceBehavioralRegressor
from datasets.sequence_behavioral import create_sequence_dataloaders
from datasets.calcium import TRAINING_SESSION_IDS


def augment_sequence(neural_data, velocity_seq, augment_prob=0.5):
    """
    Data augmentation per sequenze temporali
    
    Args:
        neural_data: (batch, neurons, 60)
        velocity_seq: (batch, 60)
        augment_prob: probabilità di applicare augmentation
    
    Returns:
        neural_data, velocity_seq (augmented)
    """
    # Random noise sul segnale neurale
    if torch.rand(1).item() < augment_prob:
        noise = torch.randn_like(neural_data) * 0.05
        neural_data = neural_data + noise
    
    # Random scaling (simula variazioni di intensità segnale)
    if torch.rand(1).item() < augment_prob:
        scale = torch.rand(1).item() * 0.2 + 0.9  # [0.9, 1.1]
        neural_data = neural_data * scale
    
    return neural_data, velocity_seq


def weighted_sequence_loss(predictions, targets, threshold=1.5, weight_high=5.0):
    """
    MSE pesata che dà più importanza ai timesteps con picchi
    
    Args:
        predictions: (batch, 60) sequenze predette
        targets: (batch, 60) sequenze target
        threshold: soglia per identificare picchi (in unità normalizzate)
        weight_high: peso per timesteps con picchi
    
    Returns:
        weighted loss (scalare)
    """
    # Errori per timestep
    errors = (predictions - targets) ** 2  # (batch, 60)
    
    # Identifica timesteps con picchi (valori alti in valore assoluto)
    is_peak = torch.abs(targets) > threshold  # (batch, 60)
    
    # Pesi: weight_high per timesteps con picchi, 1.0 per baseline
    weights = torch.where(is_peak, 
                         torch.tensor(weight_high, device=targets.device),
                         torch.tensor(1.0, device=targets.device))
    
    # Loss pesata: media su tutti i timesteps
    weighted_loss = torch.mean(weights * errors)
    
    return weighted_loss


def train_epoch(model, train_loader, optimizer, criterion, device, epoch,
                use_weighted=False, peak_threshold=1.5, peak_weight=5.0):
    """Train per un epoch"""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (neural_data, velocity_seq) in enumerate(pbar):
        # Move to device
        neural_data = neural_data.to(device)  # (batch, neurons, 60)
        velocity_seq = velocity_seq.to(device)  # (batch, 60)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(neural_data)  # (batch, 60)
        
        # Loss: MSE pesata se abilitato
        if use_weighted:
            loss = weighted_sequence_loss(predictions, velocity_seq, peak_threshold, peak_weight)
        else:
            loss = criterion(predictions, velocity_seq)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Log to wandb
        wandb.log({
            'train/batch_loss': loss.item(),
            'train/step': epoch * len(train_loader) + batch_idx
        })
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, test_loader, criterion, device, epoch):
    """Valuta il modello"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Per calcolare metriche
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for neural_data, velocity_seq in tqdm(test_loader, desc="Evaluating"):
            neural_data = neural_data.to(device)
            velocity_seq = velocity_seq.to(device)
            
            # Forward
            predictions = model(neural_data)
            
            # Loss
            loss = criterion(predictions, velocity_seq)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(velocity_seq.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    
    # Concatena
    predictions_array = np.concatenate(all_predictions, axis=0)  # (N, 60)
    targets_array = np.concatenate(all_targets, axis=0)  # (N, 60)
    
    # Calcola correlazione GLOBALE (su tutti i timesteps)
    pred_flat = predictions_array.flatten()
    target_flat = targets_array.flatten()
    
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # Calcola anche correlazione PER TIMESTEP (per vedere quale timestep è più difficile)
    correlations_per_timestep = []
    for t in range(60):
        corr_t = np.corrcoef(predictions_array[:, t], targets_array[:, t])[0, 1]
        correlations_per_timestep.append(corr_t)
    
    avg_corr_per_timestep = np.mean(correlations_per_timestep)
    
    return avg_loss, correlation, avg_corr_per_timestep, predictions_array, targets_array, correlations_per_timestep


def plot_sequence_examples(predictions, targets, epoch, n_samples=6):
    """Plotta esempi di sequenze ricostruite"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Seleziona sample casuali
    indices = np.random.choice(len(predictions), min(n_samples, len(predictions)), replace=False)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        pred_seq = predictions[idx]  # (60,)
        target_seq = targets[idx]  # (60,)
        
        timesteps = np.arange(60)
        
        # Plot
        ax.plot(timesteps, target_seq, 'b-', linewidth=2, label='Target', alpha=0.8)
        ax.plot(timesteps, pred_seq, 'r--', linewidth=2, label='Prediction', alpha=0.8)
        ax.fill_between(timesteps, target_seq, pred_seq, alpha=0.2, color='gray')
        
        # Calcola metriche per questo sample
        corr = np.corrcoef(pred_seq, target_seq)[0, 1]
        mse = np.mean((pred_seq - target_seq) ** 2)
        
        # Trova picchi
        has_peak_target = np.any(np.abs(target_seq) > 2.0)
        has_peak_pred = np.any(np.abs(pred_seq) > 2.0)
        
        title = f'Sample {idx}: r={corr:.3f}, MSE={mse:.4f}'
        if has_peak_target:
            title += ' [PEAK!]'
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Velocity (normalized)', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Epoch {epoch}: Sequence Reconstruction Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_correlation_per_timestep(correlations_per_timestep, epoch):
    """Plotta correlazione per ogni timestep"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    timesteps = np.arange(60)
    
    ax.plot(timesteps, correlations_per_timestep, 'b-o', linewidth=2, markersize=4)
    ax.axhline(y=np.mean(correlations_per_timestep), color='r', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(correlations_per_timestep):.3f}')
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title(f'Epoch {epoch}: Correlation per Timestep', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    return fig


def main():
    # ========================================================================
    # CONFIG
    # ========================================================================
    
    config = {
        # Data
        'session_id': TRAINING_SESSION_IDS[0],
        'window_size': 60,
        'stride': 30,
        'test_split': 0.2,
        'normalize': True,
        
        # Model
        'hidden_dim': 256,
        'dropout_rate': 0.3,
        'normalize_input': True,
        
        # Training
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-4,        # ✅ LR originale
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        
        # Loss - NO weighted (standard MSE funzionava meglio!)
        'use_weighted_loss': False,   # ❌ Disabilitata
        'peak_threshold': 2.0,
        'peak_weight': 3.0,
        
        # Data Augmentation - NO
        'use_augmentation': False,
        'augment_prob': 0.0,
        
        # Paths
        'save_dir': r'C:\Users\Claudio\Desktop\TesiProgettoVQVAE\results\sequence_model',
        
        # System
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 0,
        'seed': 42
    }
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Save directory: {save_dir}")
    
    # Device
    device = torch.device(config['device'])
    print(f"🖥️  Device: {device}")
    
    # ========================================================================
    # WANDB INIT
    # ========================================================================
    
    wandb.init(
        project="ricostruzioni",
        name=f"sequence_model_session_{config['session_id']}",
        config=config,
        tags=['sequence_to_sequence', 'temporal_prediction', 'no_mean']
    )
    
    print(f"\n✅ W&B initialized: {wandb.run.name}")
    
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
    print(f"   Session: {config['session_id']}")
    print(f"   Train samples: {dataset_info['train_samples']}")
    print(f"   Test samples: {dataset_info['test_samples']}")
    print(f"   Neural shape: {dataset_info['neural_shape']}")
    print(f"   Velocity shape: {dataset_info['velocity_shape']} ← FULL SEQUENCE!")
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    
    print("\n" + "="*70)
    print("🧠 CREATING MODEL")
    print("="*70)
    
    model = SequenceBehavioralRegressor(
        num_neurons=None,
        hidden_dim=config['hidden_dim'],
        dropout_rate=config['dropout_rate'],
        normalize_input=config['normalize_input'],
        output_length=config['window_size']
    )
    
    model = model.to(device)
    
    # Dummy forward pass
    print(f"\n🔨 Building model with dummy forward pass...")
    dummy_batch = next(iter(train_loader))
    dummy_neural = dummy_batch[0].to(device)
    
    print(f"   Dummy batch shape: {dummy_neural.shape}")
    print(f"   Number of neurons: {dummy_neural.shape[1]}")
    
    with torch.no_grad():
        _ = model(dummy_neural)
    
    print(f"✅ Model built successfully!")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    wandb.config.update({
        'num_neurons': dummy_neural.shape[1],
        'train_samples': dataset_info['train_samples'],
        'test_samples': dataset_info['test_samples'],
        'total_params': total_params,
        'trainable_params': trainable_params
    })
    
    # ========================================================================
    # TRAINING SETUP
    # ========================================================================
    
    criterion = nn.MSELoss()  # MSE su sequenze temporali
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    print(f"\n✅ Training setup complete")
    print(f"   Optimizer: AdamW (lr={config['learning_rate']})")
    print(f"   Scheduler: ReduceLROnPlateau")
    if config['use_weighted_loss']:
        print(f"   Loss: Weighted MSE on sequences (threshold={config['peak_threshold']}, weight={config['peak_weight']}x)")
        print(f"   → Timesteps with peaks weighted {config['peak_weight']}x more!")
    else:
        print(f"   Loss: Standard MSE on sequences")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    
    best_correlation = -np.inf
    best_epoch = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            use_weighted=config['use_weighted_loss'],
            peak_threshold=config['peak_threshold'],
            peak_weight=config['peak_weight']
        )
        
        # Evaluate
        test_loss, correlation, avg_corr_per_timestep, predictions, targets, correlations_per_timestep = evaluate(
            model, test_loader, criterion, device, epoch
        )
        
        # Scheduler step
        scheduler.step(test_loss)
        
        # Print stats
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Test Loss:  {test_loss:.6f}")
        print(f"   Global Correlation: {correlation:.4f}")
        print(f"   Avg Corr per Timestep: {avg_corr_per_timestep:.4f}")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'test/loss': test_loss,
            'test/global_correlation': correlation,
            'test/avg_corr_per_timestep': avg_corr_per_timestep,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Plot examples ogni 10 epochs
        if epoch % 10 == 0:
            # 1. Esempi di sequenze
            fig_examples = plot_sequence_examples(predictions, targets, epoch, n_samples=6)
            
            # 2. Correlazione per timestep
            fig_corr_timestep = plot_correlation_per_timestep(correlations_per_timestep, epoch)
            
            wandb.log({
                'sequence_examples': wandb.Image(fig_examples),
                'correlation_per_timestep': wandb.Image(fig_corr_timestep),
                'epoch': epoch
            })
            
            plt.close(fig_examples)
            plt.close(fig_corr_timestep)
        
        # Save best model
        if correlation > best_correlation:
            best_correlation = correlation
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_correlation': best_correlation,
                'avg_corr_per_timestep': avg_corr_per_timestep,
                'config': config
            }
            
            checkpoint_path = save_dir / f'best_sequence_model_session_{config["session_id"]}.pt'
            torch.save(checkpoint, checkpoint_path)
            
            print(f"\n✅ New best model saved! (correlation: {best_correlation:.4f})")
            print(f"   Saved to: {checkpoint_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE")
    print("="*70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best global correlation: {best_correlation:.4f}")
    
    final_model_path = save_dir / f'best_sequence_model_session_{config["session_id"]}.pt'
    
    print(f"\n💾 Model saved to:")
    print(f"   {final_model_path}")
    
    wandb.run.summary['best_epoch'] = best_epoch
    wandb.run.summary['best_correlation'] = best_correlation
    wandb.run.summary['save_path'] = str(final_model_path)
    
    wandb.finish()
    
    print(f"\n✅ Training complete! Best model saved.")
    print(f"📊 Check W&B dashboard: {wandb.run.url}")


if __name__ == "__main__":
    main()