#!/usr/bin/env python3
"""
Script per generare TUTTE le ricostruzioni del modello VQ-VAE
- Carica i pesi del modello migliore
- Produce ricostruzioni su train e test set completi
- Salva visualizzazioni separate per train/test
- Calcola metriche dettagliate
- LOG SU WANDB: Tutte le metriche e heatmap
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from pathlib import Path
import wandb

from models.vqvae import CalciumVQVAE
from datasets.calcium import SimpleAllenBrainDataset
from torch.utils.data import DataLoader

# ============================================================================
# CONFIGURAZIONE W&B
# ============================================================================
WANDB_PROJECT = "calcium-vqvae-reconstruction"
WANDB_RUN_NAME = "full-dataset-evaluation"

def load_model_from_checkpoint(checkpoint_path, config):
    """Carica il modello dai pesi salvati"""
    print(f"Caricando modello da: {checkpoint_path}")
    
    model = CalciumVQVAE(
        num_neurons=config['num_neurons'],
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        dropout_rate=config.get('dropout_rate', 0.3)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Pesi caricati (epoca {checkpoint.get('epoch', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("Pesi caricati")
    
    model = model.to(device)
    model.eval()
    
    return model, device

def generate_all_reconstructions(model, dataloader, device, dataset_name="Dataset"):
    """Genera ricostruzioni per tutto il dataset"""
    print(f"Generando ricostruzioni per {dataset_name}...")
    
    model.eval()
    all_originals = []
    all_reconstructions = []
    all_quantized = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            neural_data = batch.to(device)
            
            vq_loss, neural_recon, perplexity, quantized, encodings = model(neural_data)
            
            all_originals.append(neural_data.cpu().numpy())
            all_reconstructions.append(neural_recon.cpu().numpy())
            all_quantized.append(quantized.cpu().numpy())
            
            total_samples += neural_data.shape[0]
            
            if (batch_idx + 1) % 50 == 0:
                print(f"   Processati {batch_idx + 1} batch, {total_samples} campioni")
    
    originals = np.concatenate(all_originals, axis=0)
    reconstructions = np.concatenate(all_reconstructions, axis=0)
    quantized = np.concatenate(all_quantized, axis=0)
    
    print(f"{dataset_name} completo: {originals.shape[0]} campioni")
    
    return originals, reconstructions, quantized

def calculate_detailed_metrics(originals, reconstructions):
    """Calcola metriche dettagliate"""
    print("Calcolando metriche dettagliate...")
    
    global_mse = np.mean((originals - reconstructions) ** 2)
    global_mae = np.mean(np.abs(originals - reconstructions))
    
    ss_res = np.sum((originals - reconstructions) ** 2)
    ss_tot = np.sum((originals - np.mean(originals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    per_neuron_mse = []
    per_neuron_correlation = []
    per_neuron_r2 = []
    
    num_neurons = originals.shape[1]
    
    for neuron_idx in range(num_neurons):
        orig_neuron = originals[:, neuron_idx, :].flatten()
        recon_neuron = reconstructions[:, neuron_idx, :].flatten()
        
        mse = np.mean((orig_neuron - recon_neuron) ** 2)
        per_neuron_mse.append(mse)
        
        if np.std(orig_neuron) > 1e-8 and np.std(recon_neuron) > 1e-8:
            corr, _ = pearsonr(orig_neuron, recon_neuron)
            per_neuron_correlation.append(corr if np.isfinite(corr) else 0)
        else:
            per_neuron_correlation.append(0)
        
        ss_res_n = np.sum((orig_neuron - recon_neuron) ** 2)
        ss_tot_n = np.sum((orig_neuron - np.mean(orig_neuron)) ** 2)
        r2_n = 1 - (ss_res_n / ss_tot_n) if ss_tot_n > 0 else 0
        per_neuron_r2.append(r2_n)
    
    perfect_neurons = np.sum(np.array(per_neuron_correlation) > 0.99)
    excellent_neurons = np.sum(np.array(per_neuron_correlation) > 0.95)
    good_neurons = np.sum(np.array(per_neuron_correlation) > 0.8)
    acceptable_neurons = np.sum(np.array(per_neuron_correlation) > 0.5)
    
    metrics = {
        'global_mse': global_mse,
        'global_mae': global_mae,
        'global_r_squared': r_squared,
        'per_neuron_mse': per_neuron_mse,
        'per_neuron_correlation': per_neuron_correlation,
        'per_neuron_r2': per_neuron_r2,
        'mean_correlation': np.mean(per_neuron_correlation),
        'min_correlation': np.min(per_neuron_correlation),
        'max_correlation': np.max(per_neuron_correlation),
        'std_correlation': np.std(per_neuron_correlation),
        'perfect_neurons': perfect_neurons,
        'excellent_neurons': excellent_neurons,
        'good_neurons': good_neurons,
        'acceptable_neurons': acceptable_neurons,
        'perfect_pct': (perfect_neurons / num_neurons) * 100,
        'excellent_pct': (excellent_neurons / num_neurons) * 100,
        'good_pct': (good_neurons / num_neurons) * 100,
        'acceptable_pct': (acceptable_neurons / num_neurons) * 100
    }
    
    return metrics

def log_reconstruction_heatmaps_wandb(originals, reconstructions, dataset_name, n_examples=10):
    """Logga heatmap delle ricostruzioni su W&B"""
    
    n_samples = originals.shape[0]
    example_indices = np.linspace(0, n_samples-1, n_examples, dtype=int)
    
    for i, idx in enumerate(example_indices):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'{dataset_name} - Sample {idx}', fontsize=12)
        
        neuron_vars = np.var(originals[idx], axis=1)
        top_neurons = np.argsort(neuron_vars)[-15:]
        
        # Original
        im1 = axes[0].imshow(originals[idx, top_neurons, :], 
                            aspect='auto', cmap='viridis', vmin=-2, vmax=2)
        axes[0].set_title('Original')
        axes[0].set_ylabel('Neurons')
        axes[0].set_xlabel('Time')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Reconstruction
        im2 = axes[1].imshow(reconstructions[idx, top_neurons, :], 
                            aspect='auto', cmap='viridis', vmin=-2, vmax=2)
        axes[1].set_title('Reconstruction')
        axes[1].set_xlabel('Time')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # Error
        error = np.abs(originals[idx, top_neurons, :] - reconstructions[idx, top_neurons, :])
        im3 = axes[2].imshow(error, aspect='auto', cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel('Time')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        
        wandb.log({
            f'{dataset_name}/reconstruction_sample_{i}': wandb.Image(fig)
        })
        
        plt.close(fig)

def create_comprehensive_plots(originals_train, reconstructions_train, metrics_train,
                             originals_test, reconstructions_test, metrics_test, 
                             save_dir):
    """Crea visualizzazioni e logga su W&B"""
    
    print("Creando visualizzazioni...")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # OVERVIEW COMPARISON
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('COMPLETE MODEL EVALUATION: Train vs Test', fontsize=16, fontweight='bold')
    
    # MSE Comparison
    axes[0, 0].bar(['Train', 'Test'], [metrics_train['global_mse'], metrics_test['global_mse']], 
                   color=['skyblue', 'lightcoral'])
    axes[0, 0].set_ylabel('Mean Squared Error')
    axes[0, 0].set_title('Global MSE Comparison')
    axes[0, 0].set_yscale('log')
    
    # R² Comparison  
    axes[0, 1].bar(['Train', 'Test'], [metrics_train['global_r_squared'], metrics_test['global_r_squared']], 
                   color=['skyblue', 'lightcoral'])
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('Global R² Comparison')
    axes[0, 1].set_ylim(0, 1)
    
    # Mean Correlation Comparison
    axes[0, 2].bar(['Train', 'Test'], [metrics_train['mean_correlation'], metrics_test['mean_correlation']], 
                   color=['skyblue', 'lightcoral'])
    axes[0, 2].set_ylabel('Mean Correlation')
    axes[0, 2].set_title('Mean Correlation Comparison')
    axes[0, 2].set_ylim(0, 1)
    
    # Per-neuron correlation histograms
    axes[1, 0].hist(metrics_train['per_neuron_correlation'], bins=20, alpha=0.7, 
                    label='Train', color='skyblue', density=True)
    axes[1, 0].hist(metrics_test['per_neuron_correlation'], bins=20, alpha=0.7, 
                    label='Test', color='lightcoral', density=True)
    axes[1, 0].set_xlabel('Correlation')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Per-Neuron Correlation Distribution')
    axes[1, 0].legend()
    axes[1, 0].axvline(0.8, color='green', linestyle='--')
    axes[1, 0].axvline(0.95, color='orange', linestyle='--')
    
    # Quality categories
    categories = ['Perfect\n(>0.99)', 'Excellent\n(>0.95)', 'Good\n(>0.8)', 'Acceptable\n(>0.5)']
    train_counts = [metrics_train['perfect_pct'], metrics_train['excellent_pct'], 
                   metrics_train['good_pct'], metrics_train['acceptable_pct']]
    test_counts = [metrics_test['perfect_pct'], metrics_test['excellent_pct'], 
                  metrics_test['good_pct'], metrics_test['acceptable_pct']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, train_counts, width, label='Train', color='skyblue')
    axes[1, 1].bar(x + width/2, test_counts, width, label='Test', color='lightcoral')
    axes[1, 1].set_ylabel('Percentage of Neurons')
    axes[1, 1].set_title('Neuron Quality Categories')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].legend()
    
    # Per-neuron MSE comparison
    neuron_indices = range(len(metrics_train['per_neuron_mse']))
    axes[1, 2].scatter(neuron_indices, metrics_train['per_neuron_mse'], 
                      alpha=0.6, label='Train', color='skyblue')
    axes[1, 2].scatter(neuron_indices, metrics_test['per_neuron_mse'], 
                      alpha=0.6, label='Test', color='lightcoral')
    axes[1, 2].set_xlabel('Neuron Index')
    axes[1, 2].set_ylabel('MSE')
    axes[1, 2].set_title('Per-Neuron MSE')
    axes[1, 2].set_yscale('log')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'complete_evaluation_overview.png', dpi=300, bbox_inches='tight')
    
    # LOG SU W&B
    wandb.log({"overview/comparison_plot": wandb.Image(fig)})
    
    plt.close()
    
    # LOG HEATMAPS SU W&B
    print("Loggando heatmap su W&B...")
    log_reconstruction_heatmaps_wandb(originals_train, reconstructions_train, "train", n_examples=10)
    log_reconstruction_heatmaps_wandb(originals_test, reconstructions_test, "test", n_examples=10)

def main():
    """Main function"""
    
    print("GENERAZIONE COMPLETA RICOSTRUZIONI VQ-VAE")
    print("=" * 50)
    
    CHECKPOINT_PATH = "results/best_perfect_reconstruction.pth"
    SAVE_DIR = "reconstruction_analysis"
    
    MODEL_CONFIG = {
        'num_neurons': 30,
        'num_hiddens': 512,        
        'num_residual_layers': 6,  
        'num_residual_hiddens': 256,
        'num_embeddings': 2048,
        'embedding_dim': 256,
        'commitment_cost': 0.05,
        'dropout_rate': 0.0
    }
    
    # INIZIALIZZA W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=MODEL_CONFIG
    )
    
    print(f"W&B inizializzato: {wandb.run.url}")
    
    try:
        # Carica modello
        model, device = load_model_from_checkpoint(CHECKPOINT_PATH, MODEL_CONFIG)
        print(f"Modello caricato su {device}")
        
        # Carica dataset
        print("\nCaricando dataset...")
        dataset = SimpleAllenBrainDataset(
            window_size=50, 
            stride=10, 
            min_neurons=30
        )
        
        # Split train/test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Dataset: {len(train_dataset)} train, {len(test_dataset)} test")
        
        # Genera ricostruzioni
        print("\nGenerando ricostruzioni...")
        
        originals_train, reconstructions_train, _ = generate_all_reconstructions(
            model, train_loader, device, "TRAIN SET"
        )
        
        originals_test, reconstructions_test, _ = generate_all_reconstructions(
            model, test_loader, device, "TEST SET"  
        )
        
        # Calcola metriche
        print("\nCalcolando metriche...")
        metrics_train = calculate_detailed_metrics(originals_train, reconstructions_train)
        metrics_test = calculate_detailed_metrics(originals_test, reconstructions_test)
        
        # LOG METRICHE SU W&B
        wandb.log({
            'train/global_mse': metrics_train['global_mse'],
            'train/global_r_squared': metrics_train['global_r_squared'],
            'train/mean_correlation': metrics_train['mean_correlation'],
            'train/perfect_pct': metrics_train['perfect_pct'],
            'train/excellent_pct': metrics_train['excellent_pct'],
            'test/global_mse': metrics_test['global_mse'],
            'test/global_r_squared': metrics_test['global_r_squared'],
            'test/mean_correlation': metrics_test['mean_correlation'],
            'test/perfect_pct': metrics_test['perfect_pct'],
            'test/excellent_pct': metrics_test['excellent_pct'],
        })
        
        # LOG DISTRIBUZIONI
        wandb.log({
            'train/correlation_distribution': wandb.Histogram(metrics_train['per_neuron_correlation']),
            'test/correlation_distribution': wandb.Histogram(metrics_test['per_neuron_correlation']),
        })
        
        # Crea visualizzazioni
        print("\nCreando visualizzazioni...")
        create_comprehensive_plots(
            originals_train, reconstructions_train, metrics_train,
            originals_test, reconstructions_test, metrics_test,
            SAVE_DIR
        )
        
        # Stampa risultati
        print("\n" + "="*50)
        print("RISULTATI FINALI")
        print("="*50)
        print(f"TRAIN SET:")
        print(f"   Global MSE: {metrics_train['global_mse']:.6f}")
        print(f"   Global R²:  {metrics_train['global_r_squared']:.4f}")
        print(f"   Mean Corr:  {metrics_train['mean_correlation']:.4f}")
        print(f"   Perfect:    {metrics_train['perfect_pct']:.1f}%")
        
        print(f"\nTEST SET:")  
        print(f"   Global MSE: {metrics_test['global_mse']:.6f}")
        print(f"   Global R²:  {metrics_test['global_r_squared']:.4f}")
        print(f"   Mean Corr:  {metrics_test['mean_correlation']:.4f}")
        print(f"   Perfect:    {metrics_test['perfect_pct']:.1f}%")
        
        # Verifica overfitting
        train_test_ratio = metrics_train['global_mse'] / metrics_test['global_mse']
        
        wandb.log({
            'summary/train_test_mse_ratio': train_test_ratio
        })
        
        if train_test_ratio < 0.5:
            print(f"\nPOSSIBILE UNDERFITTING")
        elif train_test_ratio > 2.0:
            print(f"\nPossibile overfitting")
        else:
            print(f"\nBuon bilanciamento Train/Test (ratio: {train_test_ratio:.2f})")
            
        print(f"\nFile salvati in: {SAVE_DIR}/")
        print(f"W&B dashboard: {wandb.run.url}")
        
    except Exception as e:
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()