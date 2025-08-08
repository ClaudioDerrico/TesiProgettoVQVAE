#!/usr/bin/env python3
import sys
import os
# Aggiungi il percorso principale al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import wandb

# Import corretti
from models.vqvae import CalciumVQVAE
from datasets.calcium import SimpleAllenBrainDataset

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    
    for batch in dataloader:
        neural_data = batch.to(device)
        
        # Forward pass
        vq_loss, recon, perplexity, _, _ = model(neural_data)
        
        # Loss calculation
        recon_loss = F.mse_loss(recon, neural_data)
        loss = recon_loss + vq_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
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

def evaluate_model(model, dataloader, device):
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
    
    # Calculate metrics
    mse = mean_squared_error(original.flatten(), reconstructed.flatten())
    mae = mean_absolute_error(original.flatten(), reconstructed.flatten())
    
    # Per-neuron correlations
    correlations = []
    for neuron_idx in range(original.shape[1]):
        orig_neuron = original[:, neuron_idx, :].flatten()
        recon_neuron = reconstructed[:, neuron_idx, :].flatten()
        corr, _ = pearsonr(orig_neuron, recon_neuron)
        correlations.append(corr if not np.isnan(corr) else 0)
    
    mean_correlation = np.mean(correlations)
    min_correlation = np.min(correlations)
    max_correlation = np.max(correlations)
    good_neurons = np.sum(np.array(correlations) > 0.5)
    
    num_batches = len(dataloader)
    return {
        'test/mse': mse,
        'test/mae': mae,
        'test/pearson_mean': mean_correlation,
        'test/pearson_min': min_correlation,
        'test/pearson_max': max_correlation,
        'test/good_neurons': good_neurons,
        'test/good_neurons_pct': (good_neurons / len(correlations)) * 100,
        'test/vq_loss': total_vq_loss / num_batches,
        'test/perplexity': total_perplexity / num_batches,
        'per_neuron_correlations': correlations
    }

def main():
    # 🎯 Initialize W&B
    wandb.init(
        project="calcium-vqvae",  # Nome del progetto
        name="calcium-reconstruction-v1",  # Nome del run
        config={
            "model_type": "CalciumVQVAE",
            "num_neurons": 30,
            "num_hiddens": 128,
            "num_residual_layers": 2,
            "num_residual_hiddens": 32,
            "num_embeddings": 512,
            "embedding_dim": 64,
            "commitment_cost": 0.25,
            "window_size": 50,
            "stride": 10,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "max_epochs": 100,
            "patience": 10
        }
    )
    
    print("🧠 Avvio training CalciumVQVAE con W&B logging...")
    
    # Setup
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
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=0)
        
        print(f"Split: {train_size} train, {test_size} test")
        
        # Model creation
        print("🏗️  Creazione modello...")
        model = CalciumVQVAE(
            num_neurons=wandb.config.num_neurons,
            num_hiddens=wandb.config.num_hiddens,
            num_residual_layers=wandb.config.num_residual_layers,
            num_residual_hiddens=wandb.config.num_residual_hiddens,
            num_embeddings=wandb.config.num_embeddings,
            embedding_dim=wandb.config.embedding_dim,
            commitment_cost=wandb.config.commitment_cost
        ).to(device)
        
        # 📊 Log model info to W&B
        num_params = sum(p.numel() for p in model.parameters())
        wandb.log({"model/num_parameters": num_params})
        print(f"Modello creato con {num_params:,} parametri")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
        
        # Training loop
        print("🚀 Inizio training...")
        best_test_mse = float('inf')
        patience_counter = 0
        
        for epoch in range(wandb.config.max_epochs):
            # Training phase
            train_metrics = train_epoch(model, train_loader, optimizer, device)
            
            # Evaluation phase (every 5 epochs)
            if epoch % 5 == 0:
                test_metrics = evaluate_model(model, test_loader, device)
                current_test_mse = test_metrics['test/mse']
                
                # 📊 Log all metrics to W&B
                all_metrics = {**train_metrics, **test_metrics, 'epoch': epoch}
                wandb.log(all_metrics)
                
                print(f"Epoch {epoch:3d}: Train Loss={train_metrics['train/total_loss']:.4f} | "
                      f"Test MSE={test_metrics['test/mse']:.4f}, "
                      f"MAE={test_metrics['test/mae']:.4f}, "
                      f"Corr={test_metrics['test/pearson_mean']:.4f}")
                
                # Early stopping
                if current_test_mse < best_test_mse:
                    best_test_mse = current_test_mse
                    patience_counter = 0
                    
                    # Save best model
                    torch.save(model.state_dict(), 'best_calcium_vqvae.pth')
                    wandb.save('best_calcium_vqvae.pth')  # Upload to W&B
                    
                    # 📊 Log best metrics
                    wandb.log({
                        'best/mse': current_test_mse,
                        'best/mae': test_metrics['test/mae'],
                        'best/pearson_mean': test_metrics['test/pearson_mean'],
                        'best/epoch': epoch
                    })
                else:
                    patience_counter += 1
                
                if patience_counter >= wandb.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                # Log only training metrics for non-evaluation epochs
                wandb.log({**train_metrics, 'epoch': epoch})
        
        # Final comprehensive evaluation
        print("\n🎯 VALUTAZIONE FINALE:")
        final_metrics = evaluate_model(model, test_loader, device)
        
        print(f"MSE finale: {final_metrics['test/mse']:.6f}")
        print(f"MAE finale: {final_metrics['test/mae']:.6f}")  
        print(f"Correlazione media: {final_metrics['test/pearson_mean']:.6f}")
        print(f"Correlazione min/max: {final_metrics['test/pearson_min']:.3f} / {final_metrics['test/pearson_max']:.3f}")
        print(f"Neuroni con corr > 0.5: {final_metrics['test/good_neurons']}/{wandb.config.num_neurons} ({final_metrics['test/good_neurons_pct']:.1f}%)")
        
        # 📊 Log final summary
        final_summary = {
            'final/mse': final_metrics['test/mse'],
            'final/mae': final_metrics['test/mae'],
            'final/pearson_mean': final_metrics['test/pearson_mean'],
            'final/pearson_min': final_metrics['test/pearson_min'],
            'final/pearson_max': final_metrics['test/pearson_max'],
            'final/good_neurons_pct': final_metrics['test/good_neurons_pct'],
        }
        wandb.log(final_summary)
        
        # 📊 Create histogram of per-neuron correlations
        wandb.log({
            "final/neuron_correlations_histogram": wandb.Histogram(final_metrics['per_neuron_correlations'])
        })
        
        # 📊 Create summary table
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Final MSE", f"{final_metrics['test/mse']:.6f}"],
                ["Final MAE", f"{final_metrics['test/mae']:.6f}"],
                ["Mean Correlation", f"{final_metrics['test/pearson_mean']:.6f}"],
                ["Min Correlation", f"{final_metrics['test/pearson_min']:.3f}"],
                ["Max Correlation", f"{final_metrics['test/pearson_max']:.3f}"],
                ["Good Neurons (%)", f"{final_metrics['test/good_neurons_pct']:.1f}%"]
            ]
        )
        wandb.log({"final/summary_table": summary_table})
        
    except Exception as e:
        print(f"❌ Errore durante il training: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 🏁 Finish W&B run
        wandb.finish()

if __name__ == "__main__":
    main()