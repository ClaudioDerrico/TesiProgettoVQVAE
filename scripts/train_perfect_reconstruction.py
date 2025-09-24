#!/usr/bin/env python3
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

from models.vqvae import CalciumVQVAE
from datasets.calcium import SimpleAllenBrainDataset

def create_enhanced_calcium_vqvae():
    """Crea un modello con capacità aumentata per perfect reconstruction"""
    return CalciumVQVAE(
        num_neurons=30,
        num_hiddens=512,        # AUMENTATO da 128
        num_residual_layers=6,  # AUMENTATO da 2  
        num_residual_hiddens=256, # AUMENTATO da 32
        num_embeddings=2048,    # AUMENTATO da 512
        embedding_dim=256,      # AUMENTATO da 64
        commitment_cost=0.05,   # RIDOTTO da 0.25
        dropout_rate=0.0        # ELIMINATO dropout
    )

def train_epoch_enhanced(model, dataloader, optimizer, device):
    """Training epoch ottimizzato per minimizzare reconstruction loss"""
    model.train()
    
    # 🔥 DISABILITA dropout esplicitamente durante training 
    # per massimizzare capacità di memorizzazione
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    
    for batch in dataloader:
        neural_data = batch.to(device)
        
        # Forward pass
        vq_loss, recon, perplexity, _, _ = model(neural_data)
        
        # 🎯 FOCUS: Pesare di più la reconstruction loss
        recon_loss = F.mse_loss(recon, neural_data)
        
        # 🔥 RIDOTTO peso VQ loss per favorire reconstruction
        loss = recon_loss + 0.01 * vq_loss  # Era + vq_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # 🚀 GRADIENTS PIÙ GRANDI per convergenza più aggressiva
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
    
    num_batches = len(dataloader)
    return {
        'train/total_loss': total_loss / num_batches,
        'train/recon_loss': total_recon_loss / num_batches,
        'train/vq_loss': total_vq_loss / num_batches,
        'train/perplexity': total_perplexity / num_batches
    }

def evaluate_model_enhanced(model, dataloader, device):
    """Valutazione con metriche dettagliate per perfect reconstruction"""
    model.eval()
    all_original = []
    all_reconstructed = []
    total_vq_loss = 0
    total_perplexity = 0
    
    with torch.no_grad():
        for batch in dataloader:
            neural_data = batch.to(device)
            vq_loss, recon, perplexity, _, _ = model(neural_data)
            
            all_original.append(neural_data.cpu().numpy())
            all_reconstructed.append(recon.cpu().numpy())
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
    
    original = np.concatenate(all_original, axis=0)
    reconstructed = np.concatenate(all_reconstructed, axis=0)
    
    # 🎯 METRICHE DETTAGLIATE per perfect reconstruction
    mse = mean_squared_error(original.flatten(), reconstructed.flatten())
    mae = mean_absolute_error(original.flatten(), reconstructed.flatten())
    
    # Per-neuron correlations
    correlations = []
    mse_per_neuron = []
    
    for neuron_idx in range(original.shape[1]):
        orig_neuron = original[:, neuron_idx, :].flatten()
        recon_neuron = reconstructed[:, neuron_idx, :].flatten()
        
        # Correlation
        corr, _ = pearsonr(orig_neuron, recon_neuron)
        correlations.append(corr if not np.isnan(corr) else 0)
        
        # MSE per neuron
        neuron_mse = mean_squared_error(orig_neuron, recon_neuron)
        mse_per_neuron.append(neuron_mse)
    
    mean_correlation = np.mean(correlations)
    min_correlation = np.min(correlations)
    max_correlation = np.max(correlations)
    perfect_neurons = np.sum(np.array(correlations) > 0.99)  # SOGLIA PIÙ ALTA
    excellent_neurons = np.sum(np.array(correlations) > 0.95)
    good_neurons = np.sum(np.array(correlations) > 0.5)
    
    # 🎯 R² (coefficient of determination) - dovrebbe essere vicino a 1.0
    ss_res = np.sum((original.flatten() - reconstructed.flatten()) ** 2)
    ss_tot = np.sum((original.flatten() - np.mean(original.flatten())) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    num_batches = len(dataloader)
    return {
        'test/mse': mse,
        'test/mae': mae,
        'test/r_squared': r_squared,
        'test/pearson_mean': mean_correlation,
        'test/pearson_min': min_correlation,
        'test/pearson_max': max_correlation,
        'test/perfect_neurons': perfect_neurons,
        'test/perfect_neurons_pct': (perfect_neurons / len(correlations)) * 100,
        'test/excellent_neurons': excellent_neurons,
        'test/excellent_neurons_pct': (excellent_neurons / len(correlations)) * 100,
        'test/good_neurons': good_neurons,
        'test/good_neurons_pct': (good_neurons / len(correlations)) * 100,
        'test/worst_neuron_mse': np.max(mse_per_neuron),
        'test/best_neuron_mse': np.min(mse_per_neuron),
        'test/vq_loss': total_vq_loss / num_batches,
        'test/perplexity': total_perplexity / num_batches,
        'per_neuron_correlations': correlations,
        'per_neuron_mse': mse_per_neuron
    }

def main():
    """Training ottimizzato per raggiungere train loss → 0"""
    
    # 🎯 W&B config per perfect reconstruction
    wandb.init(
        project="calcium-vqvae-perfect",
        name="perfect-reconstruction-v1",
        config={
            "model_type": "Enhanced_CalciumVQVAE",
            "num_neurons": 30,
            "num_hiddens": 512,           # AUMENTATO
            "num_residual_layers": 6,     # AUMENTATO 
            "num_residual_hiddens": 256,  # AUMENTATO
            "num_embeddings": 2048,       # AUMENTATO
            "embedding_dim": 256,         # AUMENTATO
            "commitment_cost": 0.05,      # RIDOTTO
            "window_size": 50,
            "stride": 10,
            "batch_size": 16,             # RIDOTTO per più update steps
            "learning_rate": 0.001,       # AUMENTATO
            "max_epochs": 200,            # PIÙ EPOCHE
            "patience": 50,               # PIÙ PAZIENZA
            "vq_weight": 0.01,            # PESO VQ RIDOTTO
            "target_train_loss": 0.01     # OBIETTIVO
        }
    )
    
    print("🎯 Training per PERFECT RECONSTRUCTION (Train Loss → 0)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Dataset loading
        print("📊 Caricamento dataset Allen Brain...")
        dataset = SimpleAllenBrainDataset(
            window_size=wandb.config.window_size, 
            stride=wandb.config.stride, 
            min_neurons=wandb.config.num_neurons
        )
        print(f"Dataset caricato: {len(dataset)} campioni")
        
        # Train/test split
        train_size = int(0.8 * len(dataset))  # PIÙ DATI PER TRAINING
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        # 🔥 BATCH SIZE RIDOTTO = più update steps
        train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=0)
        
        print(f"Split: {train_size} train, {test_size} test")
        
        # 🚀 MODELLO POTENZIATO
        print("🏗️ Creazione modello enhanced...")
        model = create_enhanced_calcium_vqvae().to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        wandb.log({"model/num_parameters": num_params})
        print(f"Modello enhanced: {num_params:,} parametri")
        
        # 🎯 OPTIMIZER PIÙ AGGRESSIVO
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=wandb.config.learning_rate,
            weight_decay=0.0001,  # Leggera regolarizzazione
            betas=(0.9, 0.99)     # Beta2 più alto per stabilità
        )
        
        # 🔥 SCHEDULER per learning rate decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=wandb.config.max_epochs, eta_min=1e-6
        )
        
        # Training loop
        print("🚀 Inizio training enhanced...")
        best_test_mse = float('inf')
        best_train_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(wandb.config.max_epochs):
            # 🎯 TRAINING ENHANCED
            train_metrics = train_epoch_enhanced(model, train_loader, optimizer, device)
            
            # Step scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Evaluation phase (ogni 5 epoche)
            if epoch % 5 == 0:
                test_metrics = evaluate_model_enhanced(model, test_loader, device)
                current_test_mse = test_metrics['test/mse']
                current_train_loss = train_metrics['train/recon_loss']
                
                # 📊 Log metrics
                all_metrics = {
                    **train_metrics, 
                    **test_metrics, 
                    'epoch': epoch,
                    'learning_rate': current_lr
                }
                wandb.log(all_metrics)

                # 🎯 CHECK OBIETTIVI
                train_perfect = current_train_loss < 0.01
                train_excellent = current_train_loss < 0.05
                
                print(f"Epoch {epoch:3d}: Train Loss={current_train_loss:.6f} {'🎯 PERFECT!' if train_perfect else '🔥 EXCELLENT!' if train_excellent else ''}")
                print(f"           Test MSE={current_test_mse:.4f}, R²={test_metrics['test/r_squared']:.4f}")
                print(f"           Correlations: Perfect={test_metrics['test/perfect_neurons_pct']:.1f}%, Excellent={test_metrics['test/excellent_neurons_pct']:.1f}%")
                print(f"           LR={current_lr:.2e}")
                
                # Log reconstructions every 20 epochs
                if epoch % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        sample_batch = next(iter(test_loader)).to(device)
                        _, recon_sample, _, _, _ = model(sample_batch)
                        log_reconstructions_enhanced(
                            sample_batch.cpu().numpy(), 
                            recon_sample.cpu().numpy(),
                            epoch
                        )
                
                # Early stopping basato su train loss
                if current_train_loss < best_train_loss:
                    best_train_loss = current_train_loss
                    patience_counter = 0
                    
                    wandb.log({
                        'best/train_loss': current_train_loss,
                        'best/test_mse': current_test_mse,
                        'best/r_squared': test_metrics['test/r_squared'],
                        'best/epoch': epoch
                    })
                    
                    # 🎯 SALVA se train loss sufficientemente bassa
                    if current_train_loss < 0.05:
                        print(f"💾 Salvando modello con train loss = {current_train_loss:.6f}")
                else:
                    patience_counter += 1
                
                # 🎯 STOP se raggiungiamo l'obiettivo
                if current_train_loss < wandb.config.target_train_loss:
                    print(f"🎯 OBIETTIVO RAGGIUNTO! Train loss = {current_train_loss:.6f} < {wandb.config.target_train_loss}")
                    break
                
                if patience_counter >= wandb.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                # Log solo training metrics
                wandb.log({**train_metrics, 'epoch': epoch, 'learning_rate': current_lr})
        
        # Final evaluation
        print("\n🎯 VALUTAZIONE FINALE ENHANCED:")
        final_metrics = evaluate_model_enhanced(model, test_loader, device)
        final_train_metrics = train_epoch_enhanced(model, train_loader, optimizer, device)
        
        print(f"🔥 TRAIN RECON LOSS FINALE: {final_train_metrics['train/recon_loss']:.8f}")
        print(f"📊 Test MSE: {final_metrics['test/mse']:.6f}")
        print(f"📈 R² Score: {final_metrics['test/r_squared']:.6f}")
        print(f"🎯 Perfect correlations (>0.99): {final_metrics['test/perfect_neurons_pct']:.1f}%")
        print(f"⭐ Excellent correlations (>0.95): {final_metrics['test/excellent_neurons_pct']:.1f}%")
        
        # 📊 Final summary
        wandb.log({
            'final/train_recon_loss': final_train_metrics['train/recon_loss'],
            'final/test_mse': final_metrics['test/mse'],
            'final/r_squared': final_metrics['test/r_squared'],
            'final/perfect_neurons_pct': final_metrics['test/perfect_neurons_pct'],
            'final/excellent_neurons_pct': final_metrics['test/excellent_neurons_pct'],
        })
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        wandb.finish()

def log_reconstructions_enhanced(original, reconstructed, epoch, num_examples=2):
    """Enhanced reconstruction logging con metriche dettagliate"""
    for i in range(min(num_examples, original.shape[0])):
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        
        # Top neurons
        neuron_vars = np.var(original[i], axis=1)
        top_neurons = np.argsort(neuron_vars)[-15:]
        
        # Original vs Reconstruction
        im1 = axes[0,0].imshow(original[i, top_neurons, :], aspect='auto', cmap='viridis')
        axes[0,0].set_title('Original')
        axes[0,0].set_ylabel('Neurons')
        
        im2 = axes[0,1].imshow(reconstructed[i, top_neurons, :], aspect='auto', cmap='viridis')
        axes[0,1].set_title('Reconstruction')
        
        # Error heatmap
        error = np.abs(original[i, top_neurons, :] - reconstructed[i, top_neurons, :])
        im3 = axes[1,0].imshow(error, aspect='auto', cmap='Reds')
        axes[1,0].set_title('Absolute Error')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Neurons')
        
        # Per-neuron correlation
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
        wandb.log({f"enhanced_reconstructions/epoch_{epoch}_example_{i}": wandb.Image(fig)})
        plt.close(fig)

if __name__ == "__main__":
    main()