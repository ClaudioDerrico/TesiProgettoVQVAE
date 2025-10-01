#!/usr/bin/env python3
"""
Script per visualizzare ricostruzioni su batch selezionati di train e test
per verificare se le performance dipendono dai campioni specifici.

Uso:
    python scripts/visualize_batch_reconstructions.py

Output:
    - Metriche calcolate su TUTTI i campioni di ogni batch
    - Visualizzazione di 10 campioni rappresentativi per batch (migliori/peggiori/medi)
    - Confronto statistico train vs test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from pathlib import Path

from models.vqvae import CalciumVQVAE
from datasets.calcium import SimpleAllenBrainDataset
from torch.utils.data import DataLoader

# ============================================================================
# CONFIGURAZIONE - MODIFICA QUESTI PARAMETRI
# ============================================================================

CHECKPOINT_PATH = "results/best_perfect_reconstruction.pth"  # Path del tuo modello
BATCH_SIZE = 32  # Dimensione batch per calcolare le metriche
NUM_SAMPLES_TO_VISUALIZE = 10  # Quanti campioni visualizzare per batch (es. 10)
NUM_BATCHES_TO_ANALYZE = 3  # Numero di batch da analizzare (3 batch = ~90 campioni)
SAVE_DIR = "batch_reconstructions_analysis"

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

# ============================================================================
# FUNZIONI HELPER
# ============================================================================

def load_trained_model(checkpoint_path, config, device):
    """Carica il modello dai pesi salvati"""
    print(f"🔄 Caricando modello da: {checkpoint_path}")
    
    model = CalciumVQVAE(
        num_neurons=config['num_neurons'],
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        dropout_rate=config.get('dropout_rate', 0.0)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Pesi caricati (epoca {checkpoint.get('epoch', 'N/A')})")
        if 'final_train_loss' in checkpoint:
            print(f"📊 Train loss: {checkpoint.get('final_train_loss', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Pesi caricati (formato diretto)")
    
    model = model.to(device)
    model.eval()
    
    return model


def compute_sample_metrics(original, reconstructed):
    """Calcola metriche per ogni singolo campione"""
    batch_size = original.shape[0]
    
    metrics = {
        'mse': [],
        'mae': [],
        'correlation': [],
        'per_neuron_corr': []
    }
    
    for i in range(batch_size):
        # MSE e MAE per campione
        sample_orig = original[i].flatten()
        sample_recon = reconstructed[i].flatten()
        
        mse = np.mean((sample_orig - sample_recon) ** 2)
        mae = np.mean(np.abs(sample_orig - sample_recon))
        
        # Correlazione globale del campione
        if np.std(sample_orig) > 1e-8 and np.std(sample_recon) > 1e-8:
            corr, _ = pearsonr(sample_orig, sample_recon)
            corr = corr if np.isfinite(corr) else 0
        else:
            corr = 0
        
        # Correlazione per neurone
        neuron_corrs = []
        for neuron_idx in range(original.shape[1]):
            orig_neuron = original[i, neuron_idx, :]
            recon_neuron = reconstructed[i, neuron_idx, :]
            
            if np.std(orig_neuron) > 1e-8 and np.std(recon_neuron) > 1e-8:
                n_corr, _ = pearsonr(orig_neuron, recon_neuron)
                neuron_corrs.append(n_corr if np.isfinite(n_corr) else 0)
            else:
                neuron_corrs.append(0)
        
        metrics['mse'].append(mse)
        metrics['mae'].append(mae)
        metrics['correlation'].append(corr)
        metrics['per_neuron_corr'].append(np.mean(neuron_corrs))
    
    return metrics


def visualize_batch_sample(original, reconstructed, metrics, batch_idx, dataset_name, 
                          save_dir, num_samples=10):
    """Visualizza un SOTTOINSIEME rappresentativo di campioni del batch"""
    
    batch_size = original.shape[0]
    
    # Seleziona campioni rappresentativi (non solo i primi!)
    if batch_size <= num_samples:
        # Se il batch è piccolo, usa tutti
        sample_indices = list(range(batch_size))
    else:
        # Selezione strategica:
        # - Migliori 3 campioni (correlazione più alta)
        # - Peggiori 3 campioni (correlazione più bassa)  
        # - 4 campioni random dal mezzo
        sorted_indices = np.argsort(metrics['correlation'])
        
        worst_3 = sorted_indices[:3].tolist()  # 3 peggiori
        best_3 = sorted_indices[-3:].tolist()  # 3 migliori
        
        # 4 campioni dal mezzo (escludendo estremi)
        middle_pool = sorted_indices[3:-3]
        if len(middle_pool) >= 4:
            middle_4 = np.random.choice(middle_pool, 4, replace=False).tolist()
        else:
            middle_4 = middle_pool.tolist()
        
        sample_indices = worst_3 + middle_4 + best_3
        sample_indices = sorted(sample_indices)  # Ordina per chiarezza
    
    num_to_show = len(sample_indices)
    
    # =================================================================
    # FIGURA: Grid di heatmap con campioni selezionati
    # =================================================================
    fig, axes = plt.subplots(num_to_show, 3, figsize=(15, 3*num_to_show))
    fig.suptitle(f'🎯 {dataset_name} - Batch {batch_idx}: {num_to_show} Campioni Rappresentativi', 
                 fontsize=14, fontweight='bold')
    
    if num_to_show == 1:
        axes = axes.reshape(1, -1)
    
    for plot_idx, sample_idx in enumerate(sample_indices):
        # Seleziona neuroni più attivi per questo campione
        neuron_vars = np.var(original[sample_idx], axis=1)
        top_neurons = np.argsort(neuron_vars)[-15:]
        
        # Originale
        ax_orig = axes[plot_idx, 0]
        im_orig = ax_orig.imshow(original[sample_idx, top_neurons, :], 
                                 aspect='auto', cmap='viridis', vmin=-2, vmax=2)
        ax_orig.set_ylabel(f'Sample {sample_idx}', fontsize=10)
        if plot_idx == 0:
            ax_orig.set_title('Original', fontsize=11)
        
        # Ricostruito
        ax_recon = axes[plot_idx, 1]
        im_recon = ax_recon.imshow(reconstructed[sample_idx, top_neurons, :], 
                                   aspect='auto', cmap='viridis', vmin=-2, vmax=2)
        if plot_idx == 0:
            ax_recon.set_title('Reconstruction', fontsize=11)
        
        # Errore assoluto
        error = np.abs(original[sample_idx, top_neurons, :] - 
                      reconstructed[sample_idx, top_neurons, :])
        ax_error = axes[plot_idx, 2]
        im_error = ax_error.imshow(error, aspect='auto', cmap='Reds', vmin=0, vmax=1)
        if plot_idx == 0:
            ax_error.set_title('Absolute Error', fontsize=11)
        
        # Aggiungi metriche come testo
        metrics_text = (f"MSE: {metrics['mse'][sample_idx]:.4f}\n"
                       f"Corr: {metrics['correlation'][sample_idx]:.3f}")
        ax_error.text(1.15, 0.5, metrics_text, transform=ax_error.transAxes,
                     fontsize=9, verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Etichette asse x solo per l'ultima riga
        if plot_idx == num_to_show - 1:
            ax_orig.set_xlabel('Time', fontsize=10)
            ax_recon.set_xlabel('Time', fontsize=10)
            ax_error.set_xlabel('Time', fontsize=10)
        else:
            ax_orig.set_xticklabels([])
            ax_recon.set_xticklabels([])
            ax_error.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{dataset_name}_batch{batch_idx}_samples.png', 
                dpi=200, bbox_inches='tight')
    plt.close()
    
    # =================================================================
    # FIGURA 2: Distribuzione delle metriche DELL'INTERO BATCH
    # =================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'📊 {dataset_name} - Batch {batch_idx}: Metriche su TUTTI i {batch_size} Campioni', 
                 fontsize=14, fontweight='bold')
    
    # MSE distribution
    axes[0, 0].hist(metrics['mse'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(metrics['mse']), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(metrics["mse"]):.4f}')
    axes[0, 0].axvline(np.median(metrics['mse']), color='orange', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(metrics["mse"]):.4f}')
    # Evidenzia i campioni visualizzati
    for idx in sample_indices:
        axes[0, 0].axvline(metrics['mse'][idx], color='green', alpha=0.3, linewidth=1)
    axes[0, 0].set_xlabel('MSE')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'MSE Distribution (n={batch_size})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Correlation distribution
    axes[0, 1].hist(metrics['correlation'], bins=20, color='lightgreen', 
                    edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(metrics['correlation']), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(metrics["correlation"]):.3f}')
    axes[0, 1].axvline(np.median(metrics['correlation']), color='orange', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(metrics["correlation"]):.3f}')
    # Evidenzia i campioni visualizzati
    for idx in sample_indices:
        axes[0, 1].axvline(metrics['correlation'][idx], color='green', alpha=0.3, linewidth=1)
    axes[0, 1].set_xlabel('Correlation')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Correlation Distribution (n={batch_size})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample-wise performance (scatter plot)
    sample_ids = np.arange(batch_size)
    axes[1, 0].scatter(sample_ids, metrics['mse'], alpha=0.4, s=30, color='gray', 
                      label='All samples')
    # Evidenzia i campioni visualizzati
    axes[1, 0].scatter([sample_indices], [metrics['mse'][i] for i in sample_indices], 
                      s=100, color='red', marker='x', linewidths=2, 
                      label='Visualized samples', zorder=5)
    axes[1, 0].axhline(np.mean(metrics['mse']), color='blue', linestyle='--', 
                       label=f'Mean MSE: {np.mean(metrics["mse"]):.4f}')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Per-Sample MSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation vs MSE scatter
    axes[1, 1].scatter(metrics['correlation'], metrics['mse'], alpha=0.4, s=30, 
                      color='gray', label='All samples')
    # Evidenzia i campioni visualizzati
    axes[1, 1].scatter([metrics['correlation'][i] for i in sample_indices],
                      [metrics['mse'][i] for i in sample_indices],
                      s=100, color='red', marker='x', linewidths=2, 
                      label='Visualized samples', zorder=5)
    axes[1, 1].set_xlabel('Correlation')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('Correlation vs MSE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Statistiche aggregate
    stats_text = (
        f"INTERO BATCH (n={batch_size}):\n"
        f"MSE:  {np.mean(metrics['mse']):.4f} ± {np.std(metrics['mse']):.4f}\n"
        f"      Range: [{np.min(metrics['mse']):.4f}, {np.max(metrics['mse']):.4f}]\n"
        f"MAE:  {np.mean(metrics['mae']):.4f} ± {np.std(metrics['mae']):.4f}\n"
        f"Corr: {np.mean(metrics['correlation']):.3f} ± {np.std(metrics['correlation']):.3f}\n"
        f"      Range: [{np.min(metrics['correlation']):.3f}, {np.max(metrics['correlation']):.3f}]\n\n"
        f"CAMPIONI VISUALIZZATI (n={num_to_show}):\n"
        f"  {num_to_show//3} migliori + {num_to_show//3} peggiori + {num_to_show - 2*(num_to_show//3)} medi"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{dataset_name}_batch{batch_idx}_metrics.png', 
                dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Batch {batch_idx}: MSE={np.mean(metrics['mse']):.4f}±{np.std(metrics['mse']):.4f}, "
          f"Corr={np.mean(metrics['correlation']):.3f}±{np.std(metrics['correlation']):.3f}")
    print(f"      Visualizzati {num_to_show}/{batch_size} campioni rappresentativi")


def compare_train_test_batches(train_metrics_list, test_metrics_list, save_dir):
    """Confronta le metriche aggregate di train vs test"""
    
    # Aggrega tutte le metriche
    train_mse_all = []
    train_corr_all = []
    test_mse_all = []
    test_corr_all = []
    
    for metrics in train_metrics_list:
        train_mse_all.extend(metrics['mse'])
        train_corr_all.extend(metrics['correlation'])
    
    for metrics in test_metrics_list:
        test_mse_all.extend(metrics['mse'])
        test_corr_all.extend(metrics['correlation'])
    
    # Crea figura di confronto
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🎯 CONFRONTO TRAIN vs TEST: Variabilità tra Campioni', 
                 fontsize=16, fontweight='bold')
    
    # MSE comparison
    axes[0, 0].hist(train_mse_all, bins=30, alpha=0.6, label='Train', 
                    color='skyblue', density=True)
    axes[0, 0].hist(test_mse_all, bins=30, alpha=0.6, label='Test', 
                    color='lightcoral', density=True)
    axes[0, 0].axvline(np.mean(train_mse_all), color='blue', linestyle='--', 
                       label=f'Train Mean: {np.mean(train_mse_all):.4f}')
    axes[0, 0].axvline(np.mean(test_mse_all), color='red', linestyle='--', 
                       label=f'Test Mean: {np.mean(test_mse_all):.4f}')
    axes[0, 0].set_xlabel('MSE')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('MSE Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Correlation comparison
    axes[0, 1].hist(train_corr_all, bins=30, alpha=0.6, label='Train', 
                    color='skyblue', density=True)
    axes[0, 1].hist(test_corr_all, bins=30, alpha=0.6, label='Test', 
                    color='lightcoral', density=True)
    axes[0, 1].axvline(np.mean(train_corr_all), color='blue', linestyle='--', 
                       label=f'Train Mean: {np.mean(train_corr_all):.3f}')
    axes[0, 1].axvline(np.mean(test_corr_all), color='red', linestyle='--', 
                       label=f'Test Mean: {np.mean(test_corr_all):.3f}')
    axes[0, 1].set_xlabel('Correlation')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Correlation Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plots
    box_data_mse = [train_mse_all, test_mse_all]
    axes[1, 0].boxplot(box_data_mse, labels=['Train', 'Test'], patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('MSE Box Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    box_data_corr = [train_corr_all, test_corr_all]
    axes[1, 1].boxplot(box_data_corr, labels=['Train', 'Test'], patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].set_title('Correlation Box Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Statistiche aggregate
    stats_text = (
        f"TRAIN (n={len(train_mse_all)}):\n"
        f"  MSE:  {np.mean(train_mse_all):.4f} ± {np.std(train_mse_all):.4f}\n"
        f"  Corr: {np.mean(train_corr_all):.3f} ± {np.std(train_corr_all):.3f}\n\n"
        f"TEST (n={len(test_mse_all)}):\n"
        f"  MSE:  {np.mean(test_mse_all):.4f} ± {np.std(test_mse_all):.4f}\n"
        f"  Corr: {np.mean(test_corr_all):.3f} ± {np.std(test_corr_all):.3f}\n\n"
        f"VARIABILITY (CV = std/mean):\n"
        f"  CV MSE (Train): {np.std(train_mse_all)/np.mean(train_mse_all):.3f}\n"
        f"  CV MSE (Test):  {np.std(test_mse_all)/np.mean(test_mse_all):.3f}\n"
        f"  CV Corr (Train): {np.std(train_corr_all)/np.mean(train_corr_all):.3f}\n"
        f"  CV Corr (Test):  {np.std(test_corr_all)/np.mean(test_corr_all):.3f}"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'train_vs_test_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("📊 CONFRONTO FINALE TRAIN vs TEST")
    print("="*70)
    print(stats_text)
    print("="*70)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("🎯 VISUALIZZAZIONE BATCH COMPLETI: Train vs Test")
    print("="*70)
    print(f"📁 Modello: {CHECKPOINT_PATH}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔢 Batch analizzati: {NUM_BATCHES_TO_ANALYZE} train + {NUM_BATCHES_TO_ANALYZE} test")
    print(f"👀 Campioni visualizzati per batch: {NUM_SAMPLES_TO_VISUALIZE}")
    print(f"📊 Metriche calcolate su: TUTTI i campioni del batch")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Carica il modello
        print("\n🔄 Caricando modello...")
        model = load_trained_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        print(f"✅ Modello caricato su {device}")
        
        # 2. Carica dataset
        print("\n📊 Caricando dataset...")
        dataset = SimpleAllenBrainDataset(
            window_size=50, 
            stride=10, 
            min_neurons=30
        )
        
        # 3. Split train/test (stesso split del training!)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                 shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                shuffle=False, num_workers=0)
        
        print(f"✅ Dataset: {len(train_dataset)} train, {len(test_dataset)} test")
        print(f"   Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        
        # 4. Processa TRAIN batches
        print(f"\n🔵 Processando {NUM_BATCHES_TO_ANALYZE} batch TRAIN...")
        train_metrics_list = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= NUM_BATCHES_TO_ANALYZE:
                    break
                
                neural_data = batch.to(device)
                
                # Forward pass
                vq_loss, neural_recon, perplexity, _, _ = model(neural_data)
                
                # Converti in numpy
                original = neural_data.cpu().numpy()
                reconstructed = neural_recon.cpu().numpy()
                
                # Calcola metriche SU TUTTI i campioni del batch
                metrics = compute_sample_metrics(original, reconstructed)
                train_metrics_list.append(metrics)
                
                # Visualizza solo un sottoinsieme rappresentativo
                visualize_batch_sample(original, reconstructed, metrics, 
                                      batch_idx, "TRAIN", save_dir, 
                                      num_samples=NUM_SAMPLES_TO_VISUALIZE)
        
        # 5. Processa TEST batches
        print(f"\n🔴 Processando {NUM_BATCHES_TO_ANALYZE} batch TEST...")
        test_metrics_list = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= NUM_BATCHES_TO_ANALYZE:
                    break
                
                neural_data = batch.to(device)
                
                # Forward pass
                vq_loss, neural_recon, perplexity, _, _ = model(neural_data)
                
                # Converti in numpy
                original = neural_data.cpu().numpy()
                reconstructed = neural_recon.cpu().numpy()
                
                # Calcola metriche SU TUTTI i campioni del batch
                metrics = compute_sample_metrics(original, reconstructed)
                test_metrics_list.append(metrics)
                
                # Visualizza solo un sottoinsieme rappresentativo
                visualize_batch_sample(original, reconstructed, metrics, 
                                      batch_idx, "TEST", save_dir, 
                                      num_samples=NUM_SAMPLES_TO_VISUALIZE)
        
        # 6. Confronta train vs test
        print("\n📊 Creando confronto Train vs Test...")
        compare_train_test_batches(train_metrics_list, test_metrics_list, save_dir)
        
        print(f"\n✅ ANALISI COMPLETATA!")
        print(f"📁 Tutti i file salvati in: {save_dir}/")
        print(f"\n📋 File generati:")
        print(f"   Per ogni batch (0-{NUM_BATCHES_TO_ANALYZE-1}):")
        print(f"      - [TRAIN/TEST]_batch[N]_samples.png → {NUM_SAMPLES_TO_VISUALIZE} campioni rappresentativi")
        print(f"      - [TRAIN/TEST]_batch[N]_metrics.png → distribuzione metriche su tutti i {BATCH_SIZE} campioni")
        print(f"   - train_vs_test_comparison.png → confronto aggregato finale")
        
        total_analyzed = NUM_BATCHES_TO_ANALYZE * BATCH_SIZE * 2
        total_visualized = NUM_BATCHES_TO_ANALYZE * NUM_SAMPLES_TO_VISUALIZE * 2
        
        print(f"\n📊 RIEPILOGO:")
        print(f"   Campioni analizzati (metriche): {total_analyzed} (~{NUM_BATCHES_TO_ANALYZE*BATCH_SIZE} train + {NUM_BATCHES_TO_ANALYZE*BATCH_SIZE} test)")
        print(f"   Campioni visualizzati (immagini): {total_visualized} ({NUM_BATCHES_TO_ANALYZE*NUM_SAMPLES_TO_VISUALIZE} train + {NUM_BATCHES_TO_ANALYZE*NUM_SAMPLES_TO_VISUALIZE} test)")
        
        print("\n💡 COME INTERPRETARE:")
        print("   1. Guarda gli istogrammi → capire la distribuzione delle performance")
        print("   2. Confronta train vs test → verificare generalizzazione")
        print("   3. Guarda deviazione standard → se alta = performance dipendono dal campione")
        print("   4. Nei plot individuali → i migliori/peggiori/medi mostrano la variabilità")
        
    except FileNotFoundError:
        print(f"❌ File non trovato: {CHECKPOINT_PATH}")
        print("   Verifica il percorso del checkpoint!")
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()