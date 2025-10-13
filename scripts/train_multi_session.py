#!/usr/bin/env python3
"""
Training script per VQ-VAE su MULTI-SESSIONE con nuovi parametri:
- 10 sessioni di training (NUOVE, mai usate)
- 3 sessioni di test cross-session (quelle precedenti)
- Window size: 30 neuroni x 60 timesteps (era 50)
- Stride: 50 timesteps (era 10)
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

MODEL_CONFIG = {
    'num_neurons': 30,
    'num_hiddens': 512,
    'num_residual_layers': 6,
    'num_residual_hiddens': 256,
    'num_embeddings': 2048,
    'embedding_dim': 256,
    'commitment_cost': 0.0000001,  # Molto basso
    'dropout_rate': 0.0,
    'use_quantizer': True,  # ✅ ABILITATO
}

def create_enhanced_calcium_vqvae():
    """Modello con VQ ABILITATO"""
    return CalciumVQVAE(
        num_neurons=MODEL_CONFIG['num_neurons'],
        num_hiddens=MODEL_CONFIG['num_hiddens'],
        num_residual_layers=MODEL_CONFIG['num_residual_layers'],
        num_residual_hiddens=MODEL_CONFIG['num_residual_hiddens'],
        num_embeddings=MODEL_CONFIG['num_embeddings'],
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        commitment_cost=MODEL_CONFIG['commitment_cost'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        use_quantizer=MODEL_CONFIG['use_quantizer']  # ✅ CRITICO!
    )

def train_epoch_enhanced(model, dataloader, optimizer, device, epoch, config):
    """Training con VQ SCHEDULATO"""
    model.train()
    
    # Disabilita dropout
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    
    # 🔥 CALCOLA PESO VQ DINAMICO
    warmup_epochs = config.get('warmup_epochs', 10)
    max_epochs = config.get('max_epochs', 100)
    vq_weight_final = config.get('vq_weight_final', 0.01)
    
    if epoch < warmup_epochs:
        vq_weight = 0.0  # NO VQ durante warmup
        phase = "WARMUP"
    else:
        # Aumenta linearmente da 0 a vq_weight_final
        progress = (epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
        vq_weight = min(progress * vq_weight_final, vq_weight_final)
        phase = "VQ_ACTIVE"
    
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    
    for neural_data in dataloader:
        neural_data = neural_data.to(device)
        
        vq_loss, recon, perplexity, _, _ = model(neural_data)
        
        # ✅ HUBER LOSS (identico a prima)
        diff = torch.abs(recon - neural_data)
        delta = 1.0
        recon_loss = torch.where(
            diff < delta,
            0.5 * diff ** 2,
            delta * (diff - 0.5 * delta)
        ).mean()
        
        # ✅ VARIANCE LOSS (identico a prima)
        var_original = torch.var(neural_data)
        var_recon = torch.var(recon)
        var_loss = (var_original - var_recon) ** 2
        
        # 🔥 LOSS COMBINATA con VQ schedulato
        loss = recon_loss + 1.0 * var_loss + vq_weight * vq_loss
        
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
        'train/vq_weight': vq_weight,  # 🆕 Log del peso
        'train/phase': phase  # 🆕 Log della fase
    }

def evaluate_model_enhanced(model, dataloader, device):
    """Valutazione PULITA - no maschere"""
    model.eval()
    all_original = []
    all_reconstructed = []
    total_vq_loss = 0
    total_perplexity = 0
    num_batches = 0
    
    with torch.no_grad():
        for neural_data in dataloader:  # ✅ SEMPLICE! Solo dati
            neural_data = neural_data.to(device)
            
            vq_loss, recon, perplexity, _, _ = model(neural_data)
            
            all_original.append(neural_data.cpu().numpy())
            all_reconstructed.append(recon.cpu().numpy())
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
    
    original = np.concatenate(all_original, axis=0)
    reconstructed = np.concatenate(all_reconstructed, axis=0)
    
    # ✅ CALCOLO MSE GLOBALE
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    
    # ✅ R² CALCOLATO CORRETTAMENTE
    y_true = original.flatten()
    y_pred = reconstructed.flatten()
    y_mean = np.mean(y_true)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    if ss_tot < 1e-10:
        r_squared = 0.0
        print("⚠️ SS_tot troppo piccolo, R² impostato a 0")
    else:
        r_squared = 1 - (ss_res / ss_tot)

        # In evaluate_model_enhanced, dopo il calcolo di R²:
        print(f"\n🔍 DEBUG Varianza:")
        print(f"   Original mean: {y_mean:.6f}")
        print(f"   Original std: {np.std(y_true):.6f}")
        print(f"   Original min/max: [{y_true.min():.3f}, {y_true.max():.3f}]")
        print(f"   Prediction mean: {np.mean(y_pred):.6f}")
        print(f"   Prediction std: {np.std(y_pred):.6f}")
        print(f"   Prediction min/max: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
        print(f"   SS_tot: {ss_tot:.6f}")
        print(f"   SS_res: {ss_res:.6f}")
    
    
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
    """Enhanced reconstruction logging - SOLO W&B, no file locali"""
    
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
        
        try:
            # 🔥 SALVA IN MEMORIA (BytesIO) invece che su disco
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            # Apri da buffer come PIL Image
            pil_image = Image.open(buf)
            
            # Log direttamente su W&B dalla memoria
            wandb.log({f"enhanced_reconstructions/epoch_{epoch}_example_{i}": 
                      wandb.Image(pil_image)})
            
            # Chiudi buffer
            buf.close()
            
        except Exception as e:
            print(f"⚠️ W&B logging failed for epoch {epoch} example {i}: {e}")
            # Non crashare, continua il training
        
        finally:
            plt.close(fig)

def main():
    wandb.init(
        project="calcium-vqvae-multi-session",
        name="10sessions-WITH-VQ-scheduled-BS16",  # 🆕 Specifico batch size
        config={
            "model_type": "CalciumVQVAE_Scheduled",
            "num_sessions_train": len(TRAINING_SESSION_IDS),
            "num_sessions_test": len(TEST_SESSION_IDS),
            "num_neurons": 30,
            "neuron_selection": "random",
            "window_size": 60,
            "stride": 50,
            "num_hiddens": 512,
            "num_residual_layers": 6,
            "num_residual_hiddens": 256,
            "num_embeddings": 2048,
            "embedding_dim": 256,
            "commitment_cost": 0.0000001,
            "dropout_rate": 0.0,
            
            # Training parameters
            "batch_size": 16,  # ✅ PICCOLO per più update steps
            "learning_rate": 0.0003,  # OK per batch 16
            "max_epochs": 200,  # Più epoche = va bene con batch piccolo
            "patience": 50,
            
            # VQ SCHEDULING
            "warmup_epochs": 10,      # Epoch senza VQ
            "vq_weight_final": 0.01,  # Peso massimo VQ
            "use_quantizer": True,
            "target_train_loss": 0.005,
        }
    )
    
    print("🎯 TRAINING MULTI-SESSION VQ-VAE - BIG MODEL")
    print("="*70)
    print(f"📊 Training sessions: {len(TRAINING_SESSION_IDS)}")
    print(f"   {TRAINING_SESSION_IDS}")
    print(f"📐 Window: 30 neurons x 60 timesteps, stride=50")
    print(f"🏗️  Model: {wandb.config.num_hiddens}H, {wandb.config.num_residual_layers}L")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"💻 Using device: {device}")
    
    try:
        # Dataset loading
        # NUOVO (solo training sessions con split interno):
        train_loader, test_loader, dataset_info = create_unified_dataloaders(
            train_session_ids=TRAINING_SESSION_IDS,  # Solo queste
            batch_size=wandb.config.batch_size,
            test_split=0.2,  # ✅ 80% train, 20% test (stesso dataset)
            window_size=wandb.config.window_size,
            stride=wandb.config.stride,
            min_neurons=wandb.config.num_neurons,
            num_workers=0,
            use_cross_session_test=False  # ✅ Split interno
        )

        # 🔍 DEBUG: Verifica dataset
        print(f"\n🔍 DEBUG DATASET:")
        print(f"   Total samples: {dataset_info['total_samples']}")
        print(f"   Train samples: {dataset_info['train_samples']}")
        print(f"   Test samples: {dataset_info['test_samples']}")
        print(f"   Num sessions (train): {dataset_info['num_sessions_train']}")

        # 🔍 Ispeziona un batch
        sample_batch = next(iter(train_loader))
        if isinstance(sample_batch, (list, tuple)):
            print(f"\n🔍 DEBUG BATCH:")
            print(f"   Batch type: tuple/list con {len(sample_batch)} elementi")
        else:
            print(f"\n🔍 DEBUG BATCH:")
            print(f"   Batch shape: {sample_batch.shape}")

        # SOSTITUISCI CON:
        print(f"✅ Dataset multi-session caricato:")
        print(f"   Sessioni training: {dataset_info['num_sessions_train']}")  # ✅ CORRETTO
        print(f"   Train samples: {dataset_info['train_samples']}")
        print(f"   Test samples: {dataset_info['test_samples']}")
        print(f"   Shape per window: ({dataset_info['min_neurons']} neurons, {dataset_info['window_size']} timesteps)")
                
        # 🚀 MODELLO BIG
        print("\n🏗️ Creazione modello BIG...")
        model = create_enhanced_calcium_vqvae().to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        wandb.log({"model/num_parameters": num_params})
        print(f"✅ Modello BIG: {num_params:,} parametri")
        
        # 🎯 OPTIMIZER
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
        print("\n🚀 Inizio training multi-session con WARMUP...")
        best_test_mse = float('inf')
        best_train_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(wandb.config.max_epochs):
            # 🎯 TRAINING
            train_metrics = train_epoch_enhanced(
                model, train_loader, optimizer, device, 
                epoch, wandb.config  # 🆕 Passa epoch e config
            )
            
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
                
                print(f"Epoch {epoch:3d} [{train_metrics['train/phase']}]: Train Loss={current_train_loss:.6f} {'🎯 PERFECT!' if train_perfect else '🔥 EXCELLENT!' if train_excellent else ''}")
                print(f"           VQ Weight={train_metrics['train/vq_weight']:.6f}, VQ Loss={train_metrics['train/vq_loss']:.6f}")  # 🆕
                print(f"           Perplexity={train_metrics['train/perplexity']:.1f}")  # 🆕
                print(f"           Test MSE={current_test_mse:.4f}, R²={test_metrics['test/r_squared']:.4f}")
                print(f"           Correlations: Perfect={test_metrics['test/perfect_neurons_pct']:.1f}%, Excellent={test_metrics['test/excellent_neurons_pct']:.1f}%")
                print(f"           LR={current_lr:.2e}")
                
                # Log reconstructions ogni 20 epoche
                if epoch % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        batch_data = next(iter(test_loader))
                        
                        # Gestione batch
                        if isinstance(batch_data, (list, tuple)):
                            sample_batch = batch_data[0].to(device)
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
                else:
                    patience_counter += 1
                
               
                if patience_counter >= wandb.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                wandb.log({**train_metrics, 'epoch': epoch, 'learning_rate': current_lr})
        
        # Final evaluation
        print("\n🎯 VALUTAZIONE FINALE:")
        final_metrics = evaluate_model_enhanced(model, test_loader, device)
        final_train_metrics = train_epoch_enhanced(
        model, train_loader, optimizer, device,
        epoch if 'epoch' in locals() else wandb.config.max_epochs - 1,
        wandb.config
    )
        
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

        # 💾 SALVA IL MODELLO FINALE con config CORRETTI
        print(f"\n💾 Salvando modello finale...")
        os.makedirs('./results', exist_ok=True)

        final_checkpoint = {
            'epoch': epoch if 'epoch' in locals() else wandb.config.max_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': {
                'num_neurons': wandb.config.num_neurons,
                'window_size': wandb.config.window_size,
                'stride': wandb.config.stride,
                'num_hiddens': wandb.config.num_hiddens,
                'num_residual_layers': wandb.config.num_residual_layers,
                'num_residual_hiddens': wandb.config.num_residual_hiddens,
                'num_embeddings': wandb.config.num_embeddings,
                'embedding_dim': wandb.config.embedding_dim,
                'commitment_cost': wandb.config.commitment_cost,
                'dropout_rate': 0.0,
                'use_quantizer': True  # 🆕 Salva il flag
            },
            'dataset_info': dataset_info,
            'training_sessions': TRAINING_SESSION_IDS,
            'test_sessions': TEST_SESSION_IDS,
            'final_train_loss': final_train_metrics['train/recon_loss'],
            'final_test_mse': final_metrics['test/mse'],
            'final_r_squared': final_metrics['test/r_squared'],
            'final_vq_weight': final_train_metrics['train/vq_weight'],  # 🆕
            'final_perplexity': final_train_metrics['train/perplexity']  # 🆕
        }

        # ✅ SALVA UNA VOLTA SOLA con nome descrittivo
        torch.save(final_checkpoint, './results/best_multi_session_VQ_30x60.pth')
        print(f"✅ Modello salvato come ./results/best_multi_session_VQ_30x60.pth")

        # Salva anche su W&B
        try:
            wandb.save('./results/best_multi_session_VQ_30x60.pth')
            print(f"📤 Modello caricato anche su W&B")
        except Exception as e:
            print(f"⚠️ Impossibile caricare su W&B: {e}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()