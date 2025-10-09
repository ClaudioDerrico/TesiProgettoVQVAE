#!/usr/bin/env python3
"""
Training script per VQ-VAE su MULTI-SESSIONE con nuovi parametri:
- 10 sessioni di training (NUOVE, mai usate)
- 3 sessioni di test cross-session (quelle precedenti)
- Window size: 30 neuroni x 60 timesteps (era 50)
- Stride: 10 timesteps (era 50)
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

from models.vqvae import CalciumVQVAE
from datasets.calcium import (
    create_simple_calcium_dataloaders,
    TRAINING_SESSION_IDS,
    TEST_SESSION_IDS
)

def create_enhanced_calcium_vqvae():
    """Crea modello per 30 neuroni x 60 timesteps"""
    return CalciumVQVAE(
        num_neurons=30,
        num_hiddens=512,
        num_residual_layers=6,
        num_residual_hiddens=256,
        num_embeddings=2048,
        embedding_dim=256,
        commitment_cost=0.05,
        dropout_rate=0.0
    )

def train_epoch_enhanced(model, dataloader, optimizer, device):
    """Training epoch con maschere per neuroni E timesteps."""
    model.train()
    
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    
    for batch_data in dataloader:
        # 🆕 Unpacking con ENTRAMBE le maschere
        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
            neural_data, masks_time, masks_neurons = batch_data
            neural_data = neural_data.to(device)
            masks_time = masks_time.to(device)      # Shape: (batch, window_size)
            masks_neurons = masks_neurons.to(device)  # 🆕 Shape: (batch, min_neurons)
        else:
            # Fallback per compatibilità
            neural_data = batch_data[0].to(device)
            masks_time = torch.ones(neural_data.shape[0], neural_data.shape[2], 
                                   dtype=torch.bool, device=device)
            masks_neurons = torch.ones(neural_data.shape[0], neural_data.shape[1],
                                      dtype=torch.bool, device=device)
        
        vq_loss, recon, perplexity, _, _ = model(neural_data)
        
        # 🎭 Calcola loss SOLO su neuroni E timesteps validi
        # Crea maschera 2D: (batch, neurons, time)
        mask_2d = masks_neurons.unsqueeze(2) & masks_time.unsqueeze(1)
        # Shape: (batch, min_neurons, 1) & (batch, 1, window_size)
        #      → (batch, min_neurons, window_size)
        
        # Calcola MSE solo dove mask_2d=True
        squared_error = (recon - neural_data) ** 2
        masked_squared_error = squared_error * mask_2d
        
        # Media solo sui valori validi
        num_valid = mask_2d.sum()
        recon_loss = masked_squared_error.sum() / num_valid
        
        loss = recon_loss + 0.01 * vq_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
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
        'train/perplexity': total_perplexity / num_batches
    }

def evaluate_model_enhanced(model, dataloader, device):
    """Valutazione con maschere neuroni E tempo."""
    model.eval()
    all_original = []
    all_reconstructed = []
    total_vq_loss = 0
    total_perplexity = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Unpacking con ENTRAMBE le maschere
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                neural_data, masks_time, masks_neurons = batch_data
                neural_data = neural_data.to(device)
                masks_time = masks_time.to(device)
                masks_neurons = masks_neurons.to(device)
            elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                neural_data, masks_time = batch_data
                neural_data = neural_data.to(device)
                masks_time = masks_time.to(device)
                masks_neurons = torch.ones((neural_data.shape[0], neural_data.shape[1]),
                                          dtype=torch.bool, device=device)
            else:
                neural_data = batch_data.to(device)
                masks_time = torch.ones((neural_data.shape[0], neural_data.shape[2]), 
                                       dtype=torch.bool, device=device)
                masks_neurons = torch.ones((neural_data.shape[0], neural_data.shape[1]),
                                          dtype=torch.bool, device=device)
            
            vq_loss, recon, perplexity, _, _ = model(neural_data)
            
            all_original.append(neural_data.cpu().numpy())
            all_reconstructed.append(recon.cpu().numpy())
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
    
    original = np.concatenate(all_original, axis=0)
    reconstructed = np.concatenate(all_reconstructed, axis=0)
    
    # Calcola MSE globale
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    
    # Per-neuron correlations
    correlations = []
    mse_per_neuron = []
    
    for sample_idx in range(original.shape[0]):
        for neuron_idx in range(original.shape[1]):
            orig_neuron = original[sample_idx, neuron_idx, :]
            recon_neuron = reconstructed[sample_idx, neuron_idx, :]
            
            if len(orig_neuron) > 0 and np.std(orig_neuron) > 1e-8 and np.std(recon_neuron) > 1e-8:
                corr, _ = pearsonr(orig_neuron, recon_neuron)
                correlations.append(corr if not np.isnan(corr) else 0)
                
                neuron_mse = np.mean((orig_neuron - recon_neuron) ** 2)
                mse_per_neuron.append(neuron_mse)
    
    # R² score
    ss_res = np.sum((original.flatten() - reconstructed.flatten()) ** 2)
    ss_tot = np.sum((original.flatten() - np.mean(original.flatten())) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Percentuali neuroni perfetti
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
    """Enhanced reconstruction logging"""
    for i in range(min(num_examples, original.shape[0])):
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        
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

def main():
    """Main training con multi-sessione"""
    
    # 🎯 W&B config
    wandb.init(
        project="calcium-vqvae-multi-session",
        name="10sessions-random30neurons",  # ← Nuovo nome
        config={
            "model_type": "Enhanced_CalciumVQVAE",
            "num_sessions_train": len(TRAINING_SESSION_IDS),
            "num_sessions_test": len(TEST_SESSION_IDS),
            "num_neurons": 30,
            "neuron_selection": "random",  # ← NUOVO
            "window_size": 60,  # NUOVO
            "stride": 10,       # NUOVO
            "num_hiddens": 512,
            "num_residual_layers": 6,
            "num_residual_hiddens": 256,
            "num_embeddings": 2048,
            "embedding_dim": 256,
            "commitment_cost": 0.05,
            "batch_size": 16,
            "learning_rate": 0.001,
            "max_epochs": 200,
            "patience": 50,
            "vq_weight": 0.01,
            "target_train_loss": 0.01
        }
    )
    
    print("🎯 TRAINING MULTI-SESSION VQ-VAE")
    print("="*70)
    print(f"📊 Training sessions: {len(TRAINING_SESSION_IDS)}")
    print(f"   {TRAINING_SESSION_IDS}")
    print(f"📊 Test sessions (cross-session): {len(TEST_SESSION_IDS)}")
    print(f"   {TEST_SESSION_IDS}")
    print(f"📐 Window: 30 neurons x 60 timesteps, stride=10")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    
    try:
        # Dataset loading con SELEZIONE INTELLIGENTE
        print("\n📊 Caricamento dataset MULTI-SESSION con selezione casuale...")
        train_loader, test_loader, dataset_info = create_simple_calcium_dataloaders(
            batch_size=wandb.config.batch_size,
            window_size=wandb.config.window_size,
            stride=wandb.config.stride,
            min_neurons=wandb.config.num_neurons,
            use_multi_session=True,
            use_neuron_sliding=False,  # NO sliding per training
            
        )

                # 🔍 DEBUG: Verifica cosa è stato caricato
        print(f"\n🔍 DEBUG DATASET:")
        print(f"   Total samples: {dataset_info['total_samples']}")
        print(f"   Train samples: {dataset_info['train_samples']}")
        print(f"   Test samples: {dataset_info['test_samples']}")
        print(f"   Num sessions: {dataset_info['num_sessions']}")

        # 🔍 Ispeziona un batch
        sample_batch = next(iter(train_loader))
        print(f"\n🔍 DEBUG BATCH:")
        if isinstance(sample_batch, (list, tuple)):
            print(f"   Batch type: tuple/list con {len(sample_batch)} elementi")
            for i, item in enumerate(sample_batch):
                if isinstance(item, torch.Tensor):
                    print(f"   Element {i}: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"   Batch shape: {sample_batch.shape}")

        print(f"\n✅ Dataset multi-session caricato:")
        print(f"   Sessioni: {dataset_info['num_sessions']}")
        print(f"   Train samples: {dataset_info['train_samples']}")
        print(f"   Test samples: {dataset_info['test_samples']}")
        print(f"   🎲 TRAINING: Using RANDOM 30 neurons per session")
                
        # 🚀 MODELLO
        print("\n🏗️ Creazione modello enhanced...")
        model = create_enhanced_calcium_vqvae().to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        wandb.log({"model/num_parameters": num_params})
        print(f"Modello: {num_params:,} parametri")
        
        # 🎯 OPTIMIZER
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=wandb.config.learning_rate,
            weight_decay=0.0001,
            betas=(0.9, 0.99)
        )
        
        # 🔥 SCHEDULER
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=wandb.config.max_epochs, eta_min=1e-6
        )
        
        # Training loop
        print("\n🚀 Inizio training multi-session...")
        best_test_mse = float('inf')
        best_train_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(wandb.config.max_epochs):
            # 🎯 TRAINING
            train_metrics = train_epoch_enhanced(model, train_loader, optimizer, device)
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Evaluation ogni 5 epoche
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

                train_perfect = current_train_loss < 0.01
                train_excellent = current_train_loss < 0.05
                
                print(f"Epoch {epoch:3d}: Train Loss={current_train_loss:.6f} {'🎯 PERFECT!' if train_perfect else '🔥 EXCELLENT!' if train_excellent else ''}")
                print(f"           Test MSE={current_test_mse:.4f}, R²={test_metrics['test/r_squared']:.4f}")
                print(f"           Correlations: Perfect={test_metrics['test/perfect_neurons_pct']:.1f}%, Excellent={test_metrics['test/excellent_neurons_pct']:.1f}%")
                print(f"           LR={current_lr:.2e}")
                
                # Log reconstructions ogni 20 epoche
                if epoch % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        # Gestione corretta del batch con maschere
                        batch_data = next(iter(test_loader))
                        
                        # CORREGGI QUI: gestisci tutti i casi
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                            sample_batch, sample_masks_time, sample_masks_neurons = batch_data
                            sample_batch = sample_batch.to(device)
                        elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                            sample_batch, sample_masks = batch_data
                            sample_batch = sample_batch.to(device)
                        else:
                            sample_batch = batch_data.to(device)
                        
                        # Genera ricostruzioni
                        _, recon_sample, _, _, _ = model(sample_batch)
                        
                        # Log su W&B
                        log_reconstructions_enhanced(
                            sample_batch.cpu().numpy(), 
                            recon_sample.cpu().numpy(),
                            epoch
                        )
                
                # Early stopping
                if current_train_loss < best_train_loss:
                    best_train_loss = current_train_loss
                    patience_counter = 0
                    
                    wandb.log({
                        'best/train_loss': current_train_loss,
                        'best/test_mse': current_test_mse,
                        'best/r_squared': test_metrics['test/r_squared'],
                        'best/epoch': epoch
                    })
                    
                    if current_train_loss < 0.05:
                        print(f"💾 Salvando modello con train loss = {current_train_loss:.6f}")
                else:
                    patience_counter += 1
                
                # STOP se raggiungiamo l'obiettivo
                if current_train_loss < wandb.config.target_train_loss:
                    print(f"🎯 OBIETTIVO RAGGIUNTO! Train loss = {current_train_loss:.6f} < {wandb.config.target_train_loss}")
                    break
                
                if patience_counter >= wandb.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                wandb.log({**train_metrics, 'epoch': epoch, 'learning_rate': current_lr})
        
        # Final evaluation
        print("\n🎯 VALUTAZIONE FINALE:")
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

        # 💾 SALVA IL MODELLO FINALE
        print(f"\n💾 Salvando modello finale...")
        os.makedirs('./results', exist_ok=True)
        
        final_checkpoint = {
            'epoch': epoch if 'epoch' in locals() else 'final',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': {
                'num_neurons': 30,
                'window_size': 60,
                'stride': 10, # da 50 a 10 per compatibilità
                'num_hiddens': 512,
                'num_residual_layers': 6,
                'num_residual_hiddens': 256,
                'num_embeddings': 2048,
                'embedding_dim': 256,
                'commitment_cost': 0.05,
                'dropout_rate': 0.0
            },
            'dataset_info': dataset_info,
            'training_sessions': TRAINING_SESSION_IDS,
            'test_sessions': TEST_SESSION_IDS,
            'final_train_loss': final_train_metrics['train/recon_loss'],
            'final_test_mse': final_metrics['test/mse'],
            'final_r_squared': final_metrics['test/r_squared']
        }
        
        torch.save(final_checkpoint, './results/best_multi_session_30x60.pth')
        print(f"✅ Modello salvato come ./results/best_multi_session_30x60.pth")
        
        # Salva anche su W&B
        try:
            wandb.save('./results/best_multi_session_30x60.pth')
            print(f"📤 Modello caricato anche su W&B")
        except:
            print(f"⚠️ Impossibile caricare su W&B, ma file locale salvato")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()