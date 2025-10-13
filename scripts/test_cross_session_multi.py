#!/usr/bin/env python3
"""
Script per testare il modello allenato su 10 sessioni
contro le 3 sessioni di TEST CROSS-SESSION.

Modello allenato su: 10 sessioni NUOVE (TRAINING_SESSION_IDS)
Test su: 3 sessioni precedenti (TEST_SESSION_IDS)

LOG SU WANDB: Tutte le metriche e visualizzazioni

Uso:
    python scripts/test_cross_session_multi.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
from datasets.calcium import UnifiedMultiSessionDataset

from models.vqvae import CalciumVQVAE
from datasets.calcium import SimpleAllenBrainDataset, TRAINING_SESSION_IDS, TEST_SESSION_IDS

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

WANDB_PROJECT = "calcium-vqvae-cross-session-multi"
WANDB_RUN_NAME = f"test-3sessions-{datetime.now().strftime('%m%d-%H%M')}"

CHECKPOINT_PATH = "results/best_multi_session_FIXED_VQ.pth"  # ✅ Modello multi-session
SAVE_DIR = "cross_session_multi_analysis"

MODEL_CONFIG = {
    'num_neurons': 30,
    'num_hiddens': 512,        
    'num_residual_layers': 6,  
    'num_residual_hiddens': 256,
    'num_embeddings': 2048,
    'embedding_dim': 256,
    'commitment_cost': 0.05,
    'dropout_rate': 0.0,
    'use_quantizer': False,
}

BATCH_SIZE = 32
WINDOW_SIZE = 60  # NUOVO
STRIDE = 50       # NUOVO

# Full dataset analysis
USE_FULL_DATASET = True
MAX_SAMPLES = None

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
        dropout_rate=config.get('dropout_rate', 0.0),
        use_quantizer=config.get('use_quantizer', False)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Pesi caricati (epoca {checkpoint.get('epoch', 'N/A')})")
        
        if 'training_sessions' in checkpoint:
            print(f"📊 Modello allenato su {len(checkpoint['training_sessions'])} sessioni:")
            print(f"   {checkpoint['training_sessions']}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Pesi caricati")
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_session(model, session_id, device, window_size, stride, max_samples=None):
    """
    Valuta il modello su una specifica sessione con SLIDING sui neuroni
    """
    
    print(f"\n{'='*70}")
    print(f"📊 Testando sessione: {session_id}")
    print(f"{'='*70}")
    
    try:
        # Carica dataset con SLIDING sui neuroni
        dataset = UnifiedMultiSessionDataset(
        session_ids=[session_id],  # ✅ Singola sessione
        window_size=window_size,
        stride=stride,
        min_neurons=30,
        mode='test'  # ✅ Sliding su neuroni
        )
        
        print(f"   🔵 TEST MODE: Using neuron sliding window")
        print(f"   🔵 Will test on ALL neuron groups from this session")
        
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        total_samples = len(dataset)
        if max_samples:
            samples_to_process = min(max_samples, total_samples)
        else:
            samples_to_process = total_samples
        
        print(f"✅ Dataset caricato: {total_samples} campioni totali")
        print(f"📊 Analizzando: {samples_to_process} campioni")
        
        all_mse = []
        all_mae = []
        all_corr = []
        all_originals = []
        all_reconstructions = []
        
        samples_processed = 0
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                # Gestisci maschere
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                    neural_data, masks_time, masks_neurons = batch_data
                    neural_data = neural_data.to(device)
                else:
                    neural_data = batch_data.to(device)
                
                # Forward pass
                vq_loss, neural_recon, perplexity, _, _ = model(neural_data)
                
                # Converti in numpy
                original = neural_data.cpu().numpy()
                reconstructed = neural_recon.cpu().numpy()
                
                # 💾 SALVA primi 10 campioni per visualizzazione (SOLO DAL PRIMO BATCH!)
                if batch_idx == 0:
                    all_originals = original[:10]
                    all_reconstructions = reconstructed[:10]
                
                # Calcola metriche per TUTTI i campioni del batch
                for i in range(original.shape[0]):
                    if max_samples and samples_processed >= max_samples:
                        break
                    
                    sample_orig = original[i].flatten()
                    sample_recon = reconstructed[i].flatten()
                    
                    mse = np.mean((sample_orig - sample_recon) ** 2)
                    mae = np.mean(np.abs(sample_orig - sample_recon))
                    
                    if np.std(sample_orig) > 1e-8 and np.std(sample_recon) > 1e-8:
                        corr, _ = pearsonr(sample_orig, sample_recon)
                        corr = corr if np.isfinite(corr) else 0
                    else:
                        corr = 0
                    
                    all_mse.append(mse)
                    all_mae.append(mae)
                    all_corr.append(corr)
                    samples_processed += 1
                
                # Print ogni 50 batch
                if (batch_idx + 1) % 50 == 0:
                    print(f"   Processati {batch_idx + 1} batch, {samples_processed} campioni...")
                
                # Break se raggiunto il limite
                if max_samples and samples_processed >= max_samples:
                    break
        
        # Statistiche finali
        results = {
            'session_id': session_id,
            'num_samples': len(all_mse),
            'total_dataset_size': total_samples,
            'mse_mean': np.mean(all_mse),
            'mse_std': np.std(all_mse),
            'mse_median': np.median(all_mse),
            'mae_mean': np.mean(all_mae),
            'mae_std': np.std(all_mae),
            'corr_mean': np.mean(all_corr),
            'corr_std': np.std(all_corr),
            'corr_median': np.median(all_corr),
            'mse_min': np.min(all_mse),
            'mse_max': np.max(all_mse),
            'corr_min': np.min(all_corr),
            'corr_max': np.max(all_corr),
            'all_mse': all_mse,
            'all_mae': all_mae,
            'all_corr': all_corr,
            'originals': all_originals,
            'reconstructions': all_reconstructions,
            'success': True
        }
        
        print(f"\n📈 RISULTATI:")
        print(f"   Campioni analizzati: {results['num_samples']}/{total_samples}")
        print(f"   MSE:    {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
        print(f"   Corr:   {results['corr_mean']:.4f} ± {results['corr_std']:.4f}")
        
        # 📤 LOG SU W&B - Metriche
        session_label = f"test_{session_id}"
        
        wandb.log({
            f'{session_label}/mse_mean': results['mse_mean'],
            f'{session_label}/corr_mean': results['corr_mean'],
            f'{session_label}/num_samples': results['num_samples'],
        })
        
        wandb.log({
            f'{session_label}/mse_distribution': wandb.Histogram(all_mse),
            f'{session_label}/correlation_distribution': wandb.Histogram(all_corr),
        })
        
        # 📤 LOG HEATMAP SU W&B
        print(f"\n📤 Loggando heatmap su W&B per sessione {session_id}...")
        log_reconstruction_examples_wandb(
            all_originals, all_reconstructions, session_id, session_label
        )
        print(f"✅ Heatmap loggati su W&B!")
        
        return results
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'session_id': session_id,
            'success': False,
            'error': str(e)
        }


def log_reconstruction_examples_wandb(originals, reconstructions, session_id, session_label):
    """Crea e logga esempi di ricostruzione su W&B"""
    
    n_samples = min(5, len(originals))
    
    for sample_idx in range(n_samples):
        # Crea figura per questo campione
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Session {session_id} - Sample {sample_idx}', fontsize=12)
        
        # Seleziona neuroni più attivi
        neuron_vars = np.var(originals[sample_idx], axis=1)
        top_neurons = np.argsort(neuron_vars)[-15:]
        
        # Original
        im1 = axes[0].imshow(originals[sample_idx, top_neurons, :], 
                            aspect='auto', cmap='viridis', vmin=-2, vmax=2)
        axes[0].set_title('Original')
        axes[0].set_ylabel('Neurons')
        axes[0].set_xlabel('Time')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Reconstruction
        im2 = axes[1].imshow(reconstructions[sample_idx, top_neurons, :], 
                            aspect='auto', cmap='viridis', vmin=-2, vmax=2)
        axes[1].set_title('Reconstruction')
        axes[1].set_xlabel('Time')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # Error
        error = np.abs(originals[sample_idx, top_neurons, :] - 
                      reconstructions[sample_idx, top_neurons, :])
        im3 = axes[2].imshow(error, aspect='auto', cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel('Time')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        
        # Log su W&B
        wandb.log({
            f'{session_label}/reconstruction_sample_{sample_idx}': wandb.Image(fig)
        })
        
        plt.close(fig)


def visualize_cross_session_results(results_dict, save_dir):
    """Crea visualizzazioni comparative tra sessioni e logga su W&B"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Filtra solo risultati successful
    successful_results = {k: v for k, v in results_dict.items() if v['success']}
    
    if len(successful_results) < 1:
        print("⚠️ Nessuna sessione testata con successo")
        return
    
    # ========================================================================
    # FIGURA 1: Confronto metriche tra tutte le sessioni test
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🎯 CROSS-SESSION TEST: 3 Nuove Sessioni Mai Viste', 
                 fontsize=16, fontweight='bold')
    
    session_names = []
    mse_means = []
    mse_stds = []
    corr_means = []
    corr_stds = []
    
    for session_id, results in successful_results.items():
        label = f"Test {session_id}"
        session_names.append(label)
        mse_means.append(results['mse_mean'])
        mse_stds.append(results['mse_std'])
        corr_means.append(results['corr_mean'])
        corr_stds.append(results['corr_std'])
    
    # Bar plot MSE
    x_pos = np.arange(len(session_names))
    colors = ['lightcoral'] * len(session_names)  # Tutte sessioni test
    
    axes[0, 0].bar(x_pos, mse_means, yerr=mse_stds, color=colors, alpha=0.7, capsize=5)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(session_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Mean Squared Error per Test Session')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Bar plot Correlation
    axes[0, 1].bar(x_pos, corr_means, yerr=corr_stds, color=colors, alpha=0.7, capsize=5)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(session_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_title('Mean Correlation per Test Session')
    axes[0, 1].set_ylim([0.8, 1.0])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Distribution MSE
    mse_distributions = [results['all_mse'] for results in successful_results.values()]
    bp1 = axes[1, 0].boxplot(mse_distributions, labels=session_names, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    axes[1, 0].set_xticklabels(session_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('MSE Distribution per Test Session')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Distribution Correlation
    corr_distributions = [results['all_corr'] for results in successful_results.values()]
    bp2 = axes[1, 1].boxplot(corr_distributions, labels=session_names, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    axes[1, 1].set_xticklabels(session_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].set_title('Correlation Distribution per Test Session')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Aggiungi statistiche
    avg_samples = int(np.mean([v['num_samples'] for v in successful_results.values()]))
    avg_mse = np.mean([v['mse_mean'] for v in successful_results.values()])
    avg_corr = np.mean([v['corr_mean'] for v in successful_results.values()])
    
    stats_text = "GENERALIZATION TEST SUMMARY:\n\n"
    stats_text += f"Test Sessions: {len(successful_results)}\n"
    stats_text += f"Avg Samples per Session: {avg_samples}\n\n"
    stats_text += f"Average Performance:\n"
    stats_text += f"  MSE:  {avg_mse:.6f}\n"
    stats_text += f"  Corr: {avg_corr:.4f}\n\n"
    
    stats_text += "Per-Session Results:\n"
    for session_id, results in successful_results.items():
        stats_text += f"  {session_id}:\n"
        stats_text += f"    MSE:  {results['mse_mean']:.6f}\n"
        stats_text += f"    Corr: {results['corr_mean']:.4f}\n"
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Salva localmente
    plt.savefig(save_dir / 'cross_session_test_comparison.png', dpi=200, bbox_inches='tight')
    
    # LOG SU W&B
    wandb.log({
        "cross_session/comparison_plot": wandb.Image(fig)
    })
    
    plt.close()
    
    print(f"✅ Visualizzazione salvata localmente e su W&B")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("🎯 CROSS-SESSION TEST: Modello Multi-Session su 3 Sessioni Test")
    print("="*70)
    print(f"📁 Modello: {CHECKPOINT_PATH}")
    print(f"🔬 Sessioni test (mai viste durante training): {TEST_SESSION_IDS}")
    print(f"📊 Modalità: {'FULL DATASET' if USE_FULL_DATASET else f'LIMITED ({MAX_SAMPLES} samples)'}")
    print(f"📐 Parametri: window={WINDOW_SIZE}, stride={STRIDE}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # INIZIALIZZA W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "checkpoint_path": CHECKPOINT_PATH,
            "test_sessions": TEST_SESSION_IDS,
            "num_training_sessions": len(TRAINING_SESSION_IDS),
            "batch_size": BATCH_SIZE,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "use_full_dataset": USE_FULL_DATASET,
            "max_samples": MAX_SAMPLES,
            **MODEL_CONFIG
        },
        tags=["cross-session", "multi-session-training", "generalization-test"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    # Carica modello
    model = load_trained_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
    
    # Dizionario per salvare risultati
    all_results = {}
    
    # Test su 3 sessioni nuove
    print("\n" + "="*70)
    print("🔬 CROSS-SESSION TEST: Sessioni Mai Viste")
    print("="*70)
    
    max_samples = None if USE_FULL_DATASET else MAX_SAMPLES
    
    for session_id in TEST_SESSION_IDS:
        results = evaluate_session(model, session_id, device, WINDOW_SIZE, STRIDE, max_samples)
        all_results[session_id] = results
    
    # Crea visualizzazioni comparative
    print("\n" + "="*70)
    print("🎨 Creando visualizzazioni comparative...")
    print("="*70)
    visualize_cross_session_results(all_results, save_dir)
    
    # Summary finale
    print("\n" + "="*70)
    print("📊 SUMMARY FINALE")
    print("="*70)
    
    successful_count = sum(1 for r in all_results.values() if r['success'])
    print(f"✅ Sessioni testate con successo: {successful_count}/{len(all_results)}")
    
    # LOG SUMMARY SU W&B
    summary_metrics = {}
    
    test_session_results = [v for k, v in all_results.items() if v['success']]
    
    if test_session_results:
        avg_mse = np.mean([r['mse_mean'] for r in test_session_results])
        avg_corr = np.mean([r['corr_mean'] for r in test_session_results])
        avg_samples = int(np.mean([r['num_samples'] for r in test_session_results]))
        
        print(f"\n🔬 TEST SESSIONS (Average over {len(test_session_results)} sessions):")
        print(f"   Samples: {avg_samples} per session")
        print(f"   MSE:  {avg_mse:.6f}")
        print(f"   Corr: {avg_corr:.4f}")
        
        summary_metrics['test_sessions_mse_avg'] = avg_mse
        summary_metrics['test_sessions_corr_avg'] = avg_corr
        summary_metrics['test_sessions_samples_avg'] = avg_samples
        summary_metrics['num_test_sessions'] = len(test_session_results)
        
        # Valuta qualità generalizzazione
        if avg_corr > 0.95:
            generalization_quality = "Excellent"
            print(f"   ✅ Eccellente generalizzazione cross-session!")
        elif avg_corr > 0.90:
            generalization_quality = "Good"
            print(f"   ✅ Buona generalizzazione cross-session")
        elif avg_corr > 0.80:
            generalization_quality = "Moderate"
            print(f"   ⚠️  Generalizzazione moderata")
        else:
            generalization_quality = "Poor"
            print(f"   ❌ Generalizzazione scarsa")
        
        summary_metrics['generalization_quality'] = generalization_quality
    
    # LOG SUMMARY TABLE SU W&B
    summary_table_data = []
    for session_id, results in all_results.items():
        if results['success']:
            summary_table_data.append([
                session_id,
                results['num_samples'],
                results['total_dataset_size'],
                f"{100*results['num_samples']/results['total_dataset_size']:.1f}%",
                results['mse_mean'],
                results['mse_std'],
                results['corr_mean'],
                results['corr_std']
            ])
    
    summary_table = wandb.Table(
        columns=["Session ID", "Samples Analyzed", "Total Dataset", "Coverage %", 
                 "MSE Mean", "MSE Std", "Corr Mean", "Corr Std"],
        data=summary_table_data
    )
    
    wandb.log({
        "summary/test_sessions_table": summary_table,
        **summary_metrics
    })
    
    print(f"\n📁 Risultati salvati localmente in: {save_dir}/")
    print(f"🌐 Risultati salvati su W&B: {wandb.run.url}")
    print("="*70)
    
    # FINISH W&B
    wandb.finish()
    print("✅ W&B run completato!")


if __name__ == "__main__":
    main()