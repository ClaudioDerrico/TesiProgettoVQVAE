#!/usr/bin/env python3
"""
Script per generare TUTTE le ricostruzioni del modello VQ-VAE
- Carica i pesi del modello migliore
- Produce ricostruzioni su train e test set completi
- Salva visualizzazioni separate per train/test
- Calcola metriche dettagliate
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

from models.vqvae import CalciumVQVAE
from datasets.calcium import SimpleAllenBrainDataset
from torch.utils.data import DataLoader

def load_model_from_checkpoint(checkpoint_path, config):
    """Carica il modello dai pesi salvati"""
    print(f"🔄 Caricando modello da: {checkpoint_path}")
    
    # Crea il modello con la stessa configurazione
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
    
    # Carica i pesi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Pesi caricati (epoca {checkpoint.get('epoch', 'sconosciuta')})")
        print(f"📊 Best loss: {checkpoint.get('best_test_loss', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Pesi caricati (formato diretto)")
    
    model = model.to(device)
    model.eval()
    
    return model, device

def generate_all_reconstructions(model, dataloader, device, dataset_name="Dataset"):
    """Genera ricostruzioni per tutto il dataset"""
    print(f"🎯 Generando ricostruzioni per {dataset_name}...")
    
    model.eval()
    all_originals = []
    all_reconstructions = []
    all_quantized = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            neural_data = batch.to(device)
            
            # Forward pass completo
            vq_loss, neural_recon, perplexity, quantized, encodings = model(neural_data)
            
            # Salva tutti i dati
            all_originals.append(neural_data.cpu().numpy())
            all_reconstructions.append(neural_recon.cpu().numpy())
            all_quantized.append(quantized.cpu().numpy())
            
            total_samples += neural_data.shape[0]
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Processati {batch_idx + 1} batch, {total_samples} campioni totali")
    
    # Concatena tutti i risultati
    originals = np.concatenate(all_originals, axis=0)
    reconstructions = np.concatenate(all_reconstructions, axis=0)
    quantized = np.concatenate(all_quantized, axis=0)
    
    print(f"✅ {dataset_name} completo: {originals.shape[0]} campioni")
    
    return originals, reconstructions, quantized

def calculate_detailed_metrics(originals, reconstructions):
    """Calcola metriche dettagliate per le ricostruzioni"""
    print("📊 Calcolando metriche dettagliate...")
    
    # MSE globale
    global_mse = np.mean((originals - reconstructions) ** 2)
    global_mae = np.mean(np.abs(originals - reconstructions))
    
    # R² globale
    ss_res = np.sum((originals - reconstructions) ** 2)
    ss_tot = np.sum((originals - np.mean(originals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Metriche per neurone
    per_neuron_mse = []
    per_neuron_correlation = []
    per_neuron_r2 = []
    
    num_neurons = originals.shape[1]
    
    for neuron_idx in range(num_neurons):
        # Estrai dati per questo neurone (tutti i campioni, tutti i timepoints)
        orig_neuron = originals[:, neuron_idx, :].flatten()
        recon_neuron = reconstructions[:, neuron_idx, :].flatten()
        
        # MSE per neurone
        mse = np.mean((orig_neuron - recon_neuron) ** 2)
        per_neuron_mse.append(mse)
        
        # Correlazione
        if np.std(orig_neuron) > 1e-8 and np.std(recon_neuron) > 1e-8:
            corr, _ = pearsonr(orig_neuron, recon_neuron)
            per_neuron_correlation.append(corr if np.isfinite(corr) else 0)
        else:
            per_neuron_correlation.append(0)
        
        # R² per neurone
        ss_res_n = np.sum((orig_neuron - recon_neuron) ** 2)
        ss_tot_n = np.sum((orig_neuron - np.mean(orig_neuron)) ** 2)
        r2_n = 1 - (ss_res_n / ss_tot_n) if ss_tot_n > 0 else 0
        per_neuron_r2.append(r2_n)
    
    # Statistiche di qualità
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

def create_comprehensive_plots(originals_train, reconstructions_train, metrics_train,
                             originals_test, reconstructions_test, metrics_test, 
                             save_dir):
    """Crea visualizzazioni complete e separate per train/test"""
    
    print("🎨 Creando visualizzazioni complete...")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. OVERVIEW COMPARISON
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🎯 COMPLETE MODEL EVALUATION: Train vs Test', fontsize=16, fontweight='bold')
    
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
    axes[1, 0].axvline(0.8, color='green', linestyle='--', label='Good threshold')
    axes[1, 0].axvline(0.95, color='orange', linestyle='--', label='Excellent threshold')
    
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
    plt.close()
    
    # 2. DETAILED RECONSTRUCTION EXAMPLES - TRAIN
    plot_reconstruction_examples(originals_train, reconstructions_train, 
                               save_dir / 'train_reconstructions_detailed.png', 
                               "🔵 TRAIN SET - Detailed Reconstructions", 
                               metrics_train['per_neuron_correlation'])
    
    # 3. DETAILED RECONSTRUCTION EXAMPLES - TEST  
    plot_reconstruction_examples(originals_test, reconstructions_test, 
                               save_dir / 'test_reconstructions_detailed.png', 
                               "🔴 TEST SET - Detailed Reconstructions", 
                               metrics_test['per_neuron_correlation'])
    
    # 4. SUMMARY STATISTICS TABLE
    create_summary_table(metrics_train, metrics_test, save_dir)
    
    print(f"✅ Visualizzazioni salvate in: {save_dir}")

def plot_reconstruction_examples(originals, reconstructions, save_path, title, correlations, n_examples=4):
    """Crea plot dettagliati delle ricostruzioni"""
    
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4*n_examples))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Seleziona esempi rappresentativi
    n_samples = originals.shape[0]
    example_indices = np.linspace(0, n_samples-1, n_examples, dtype=int)
    
    for i, idx in enumerate(example_indices):
        # Seleziona i neuroni più attivi
        neuron_vars = np.var(originals[idx], axis=1)
        top_neurons = np.argsort(neuron_vars)[-15:]  # Top 15 neuroni
        
        # Originale
        im1 = axes[i, 0].imshow(originals[idx, top_neurons, :], 
                               aspect='auto', cmap='viridis')
        axes[i, 0].set_title(f'Original (Sample {idx})')
        if i == n_examples-1:
            axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Neurons')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
        
        # Ricostruzione
        im2 = axes[i, 1].imshow(reconstructions[idx, top_neurons, :], 
                               aspect='auto', cmap='viridis')
        axes[i, 1].set_title(f'Reconstruction')
        if i == n_examples-1:
            axes[i, 1].set_xlabel('Time')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
        
        # Errore assoluto
        error = np.abs(originals[idx, top_neurons, :] - reconstructions[idx, top_neurons, :])
        im3 = axes[i, 2].imshow(error, aspect='auto', cmap='Reds')
        axes[i, 2].set_title(f'Absolute Error')
        if i == n_examples-1:
            axes[i, 2].set_xlabel('Time')
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.046)
        
        # Aggiungi correlazione media per questo campione
        sample_corrs = []
        for nidx in top_neurons:
            orig = originals[idx, nidx, :]
            recon = reconstructions[idx, nidx, :]
            if np.std(orig) > 1e-8 and np.std(recon) > 1e-8:
                corr, _ = pearsonr(orig, recon)
                sample_corrs.append(corr if np.isfinite(corr) else 0)
        
        mean_corr = np.mean(sample_corrs) if sample_corrs else 0
        fig.text(0.02, 0.95 - i*0.25, f'Avg Corr: {mean_corr:.3f}', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(metrics_train, metrics_test, save_dir):
    """Crea tabella riassuntiva"""
    
    summary_data = {
        'Metric': [
            'Global MSE', 'Global R²', 'Mean Correlation',
            'Perfect Neurons (%)', 'Excellent Neurons (%)', 
            'Good Neurons (%)', 'Acceptable Neurons (%)'
        ],
        'Train': [
            f"{metrics_train['global_mse']:.6f}",
            f"{metrics_train['global_r_squared']:.4f}",
            f"{metrics_train['mean_correlation']:.4f}",
            f"{metrics_train['perfect_pct']:.1f}%",
            f"{metrics_train['excellent_pct']:.1f}%", 
            f"{metrics_train['good_pct']:.1f}%",
            f"{metrics_train['acceptable_pct']:.1f}%"
        ],
        'Test': [
            f"{metrics_test['global_mse']:.6f}",
            f"{metrics_test['global_r_squared']:.4f}",
            f"{metrics_test['mean_correlation']:.4f}",
            f"{metrics_test['perfect_pct']:.1f}%",
            f"{metrics_test['excellent_pct']:.1f}%",
            f"{metrics_test['good_pct']:.1f}%", 
            f"{metrics_test['acceptable_pct']:.1f}%"
        ]
    }
    
    # Salva come CSV
    import pandas as pd
    df = pd.DataFrame(summary_data)
    df.to_csv(save_dir / 'summary_metrics.csv', index=False)
    print(f"📄 Tabella riassuntiva salvata: {save_dir / 'summary_metrics.csv'}")

def main():
    """Main function - genera tutte le ricostruzioni"""
    
    print("🎯 GENERAZIONE COMPLETA RICOSTRUZIONI VQ-VAE")
    print("=" * 50)
    
    # ⚠️  MODIFICA QUESTI PERCORSI SECONDO I TUOI FILE
    CHECKPOINT_PATH = "results/best_perfect_reconstruction.pth"
    SAVE_DIR = "reconstruction_analysis"
    
    # Configurazione del modello (deve corrispondere a quello allenato)
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
    
    try:
        # 1. Carica il modello
        model, device = load_model_from_checkpoint(CHECKPOINT_PATH, MODEL_CONFIG)
        print(f"✅ Modello caricato su {device}")
        
        # 2. Carica dataset (stesso setup del training)
        print("\n📊 Caricando dataset...")
        dataset = SimpleAllenBrainDataset(
            window_size=50, 
            stride=10, 
            min_neurons=30
        )
        
        # 3. Crea split train/test (stesso split del training!)
        train_size = int(0.8 * len(dataset))  # Usa lo stesso split!
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size], 
            generator=torch.Generator().manual_seed(42)  # Seed fisso per riproducibilità
        )
        
        # 4. Crea dataloaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✅ Dataset: {len(train_dataset)} train, {len(test_dataset)} test")
        
        # 5. Genera TUTTE le ricostruzioni
        print("\n🔄 Generando ricostruzioni complete...")
        
        # TRAIN SET
        originals_train, reconstructions_train, quantized_train = generate_all_reconstructions(
            model, train_loader, device, "TRAIN SET"
        )
        
        # TEST SET
        originals_test, reconstructions_test, quantized_test = generate_all_reconstructions(
            model, test_loader, device, "TEST SET"  
        )
        
        # 6. Calcola metriche dettagliate
        print("\n📊 Calcolando metriche...")
        metrics_train = calculate_detailed_metrics(originals_train, reconstructions_train)
        metrics_test = calculate_detailed_metrics(originals_test, reconstructions_test)
        
        # 7. Crea visualizzazioni
        print("\n🎨 Creando visualizzazioni...")
        create_comprehensive_plots(
            originals_train, reconstructions_train, metrics_train,
            originals_test, reconstructions_test, metrics_test,
            SAVE_DIR
        )
        
        # 8. Stampa risultati finali
        print("\n" + "="*50)
        print("🎯 RISULTATI FINALI")
        print("="*50)
        print(f"📊 TRAIN SET:")
        print(f"   Global MSE: {metrics_train['global_mse']:.6f}")
        print(f"   Global R²:  {metrics_train['global_r_squared']:.4f}")
        print(f"   Mean Corr:  {metrics_train['mean_correlation']:.4f}")
        print(f"   Perfect:    {metrics_train['perfect_pct']:.1f}%")
        print(f"   Excellent:  {metrics_train['excellent_pct']:.1f}%")
        
        print(f"\n📊 TEST SET:")  
        print(f"   Global MSE: {metrics_test['global_mse']:.6f}")
        print(f"   Global R²:  {metrics_test['global_r_squared']:.4f}")
        print(f"   Mean Corr:  {metrics_test['mean_correlation']:.4f}")
        print(f"   Perfect:    {metrics_test['perfect_pct']:.1f}%")
        print(f"   Excellent:  {metrics_test['excellent_pct']:.1f}%")
        
        # 9. Verifica overfitting
        train_test_diff = metrics_train['global_mse'] / metrics_test['global_mse']
        if train_test_diff < 0.5:
            print(f"\n🚨 POSSIBILE UNDERFITTING (Train MSE molto più basso di Test)")
        elif train_test_diff > 2.0:
            print(f"\n⚠️  Possibile overfitting leggero (Train MSE: {metrics_train['global_mse']:.6f} vs Test MSE: {metrics_test['global_mse']:.6f})")
        else:
            print(f"\n✅ Buon bilanciamento Train/Test (ratio MSE: {train_test_diff:.2f})")
            
        print(f"\n📁 Tutti i file salvati in: {SAVE_DIR}/")
        print("   - complete_evaluation_overview.png (confronto train/test)")
        print("   - train_reconstructions_detailed.png (esempi train)")  
        print("   - test_reconstructions_detailed.png (esempi test)")
        print("   - summary_metrics.csv (tabella riassuntiva)")
        
    except FileNotFoundError:
        print(f"❌ File non trovato: {CHECKPOINT_PATH}")
        print("   Verifica il percorso del checkpoint!")
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()