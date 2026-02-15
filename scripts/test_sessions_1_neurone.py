#!/usr/bin/env python3
"""
Script per testare il modello SINGLE NEURON su sessioni cross-session.

Modello allenato su: 10 sessioni (1 neurone per sessione)
Test su: 9 sessioni diverse (TEST_SESSION_IDS)

Visualizzazioni: TRACCE TEMPORALI (non heatmap)

Uso:
    python scripts/test_cross_session_single_neuron.py
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
from datasets.calcium import TRAINING_SESSION_IDS, TEST_SESSION_IDS

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

WANDB_PROJECT = "calcium-vqvae-cross-session-single-neuron"
WANDB_RUN_NAME = f"test-single-neuron-1024_pt2"

CHECKPOINT_PATH = "results/best_single_neuron_model_1024_pt2.pth"  # ✅ Modello single neuron
SAVE_DIR = "cross_session_single_neuron_analysis"

MODEL_CONFIG = {
    'num_neurons': 1,           
    'num_hiddens': 128,         # ← AUMENTATO (era 64)
    'num_residual_layers': 3,   # ← AUMENTATO (era 2)
    'num_residual_hiddens': 64, # ← AUMENTATO (era 32)
    'num_embeddings': 1024,       # ← AUMENTATO (era 16)
    'embedding_dim': 128,        # ← AUMENTATO per embeddings 256(era 32)
    'commitment_cost': 0.5,    # ← aumentato per 256 (era 0.25)
    'dropout_rate': 0.0,        # ← AGGIUNTO dropout leggero
    'use_quantizer': True,
}

BATCH_SIZE = 64
WINDOW_SIZE = 60
STRIDE = 50

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
        use_quantizer=config.get('use_quantizer', True)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Pesi caricati")
        
        if 'best_correlation' in checkpoint:
            print(f"📊 Best training correlation: {checkpoint['best_correlation']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Pesi caricati")
    
    model = model.to(device)
    model.eval()
    
    return model


def visualize_single_neuron_traces(original, reconstructed, session_id, sample_indices):
    """
    Visualizza tracce del singolo neurone (come in train_di_1_neurone.py)
    
    Args:
        original: (n_samples, 1, 60)
        reconstructed: (n_samples, 1, 60)
        session_id: ID della sessione
        sample_indices: indici dei sample da visualizzare
    """
    
    num_examples = len(sample_indices)
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(18, 3*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for plot_idx, sample_idx in enumerate(sample_indices):
        # Get single neuron trace
        orig_trace = original[sample_idx, 0, :]  # Shape: (60,)
        recon_trace = reconstructed[sample_idx, 0, :]
        
        time_points = np.arange(len(orig_trace))
        
        # Calcola metriche
        amplitude = np.max(orig_trace) - np.min(orig_trace)
        
        # Identifica picchi
        mean_val = np.mean(orig_trace)
        std_val = np.std(orig_trace)
        threshold = mean_val + 2 * std_val
        peak_mask = orig_trace > threshold
        num_peaks = np.sum(np.diff(peak_mask.astype(int)) == 1)
        max_abs_value = np.max(np.abs(orig_trace))
        
        # === Plot 1: Overlay con highlight dei picchi ===
        axes[plot_idx, 0].plot(time_points, orig_trace, 'b-', 
                               label='Original', alpha=0.7, linewidth=1.5)
        axes[plot_idx, 0].plot(time_points, recon_trace, 'r-', 
                               label='Reconstruction', alpha=0.7, linewidth=1.5)
        
        # Evidenzia zona dei picchi
        if num_peaks > 0:
            axes[plot_idx, 0].fill_between(time_points, threshold, np.max(orig_trace), 
                                    alpha=0.1, color='yellow', label=f'Peak zone')
            
            # Marca i picchi individuali
            peak_indices = np.where(peak_mask)[0]
            if len(peak_indices) > 0:
                axes[plot_idx, 0].scatter(peak_indices, orig_trace[peak_indices], 
                                 color='blue', s=30, zorder=5, alpha=0.8, 
                                 marker='^', label='Peaks (orig)')
                axes[plot_idx, 0].scatter(peak_indices, recon_trace[peak_indices], 
                                 color='red', s=30, zorder=5, alpha=0.8, 
                                 marker='v', label='Peaks (recon)')
        
        axes[plot_idx, 0].set_xlabel('Time')
        axes[plot_idx, 0].set_ylabel('Activity')
        axes[plot_idx, 0].set_title(f'Sample {sample_idx}: Peaks={num_peaks}, '
                                     f'Amp={amplitude:.2f}, Max={max_abs_value:.2f}')
        axes[plot_idx, 0].legend(loc='upper right', fontsize=8)
        axes[plot_idx, 0].grid(True, alpha=0.3)
        
        # === Plot 2: Error con focus sui picchi ===
        error = np.abs(orig_trace - recon_trace)
        axes[plot_idx, 1].plot(time_points, error, 'k-', alpha=0.7, linewidth=1)
        axes[plot_idx, 1].fill_between(time_points, 0, error, alpha=0.3, color='red')
        
        # Evidenzia errore nelle zone dei picchi
        if num_peaks > 0:
            peak_error = error.copy()
            peak_error[~peak_mask] = 0
            axes[plot_idx, 1].fill_between(time_points, 0, peak_error, alpha=0.6, 
                                   color='darkred', label='Peak error')
            
            mean_peak_err = np.mean(error[peak_mask]) if np.sum(peak_mask) > 0 else 0
            axes[plot_idx, 1].set_title(f'Sample {sample_idx}: Error '
                                        f'(Peak err: {mean_peak_err:.3f})')
        else:
            axes[plot_idx, 1].set_title(f'Sample {sample_idx}: Error (No peaks)')
        
        axes[plot_idx, 1].set_xlabel('Time')
        axes[plot_idx, 1].set_ylabel('Absolute Error')
        if num_peaks > 0:
            axes[plot_idx, 1].legend(loc='upper right', fontsize=8)
        axes[plot_idx, 1].grid(True, alpha=0.3)
        
        # === Plot 3: Statistiche e correlazione ===
        if np.std(orig_trace) > 1e-8 and np.std(recon_trace) > 1e-8:
            corr, _ = pearsonr(orig_trace, recon_trace)
            
            # Correlazione solo sui picchi
            if np.sum(peak_mask) >= 2:
                peak_corr, _ = pearsonr(orig_trace[peak_mask], recon_trace[peak_mask])
            else:
                peak_corr = np.nan
        else:
            corr = 0
            peak_corr = np.nan
        
        # Scatter plot: Original vs Reconstructed
        axes[plot_idx, 2].scatter(orig_trace, recon_trace, alpha=0.5, s=20)
        
        # Linea identità
        min_val = min(np.min(orig_trace), np.min(recon_trace))
        max_val = max(np.max(orig_trace), np.max(recon_trace))
        axes[plot_idx, 2].plot([min_val, max_val], [min_val, max_val], 
                               'r--', alpha=0.5, label='Perfect reconstruction')
        
        axes[plot_idx, 2].set_xlabel('Original Activity')
        axes[plot_idx, 2].set_ylabel('Reconstructed Activity')
        axes[plot_idx, 2].set_title(f'Sample {sample_idx}: Correlation')
        axes[plot_idx, 2].legend(loc='upper left', fontsize=8)
        axes[plot_idx, 2].grid(True, alpha=0.3)
        
        # Box con statistiche
        stats_text = f'Overall Corr: {corr:.3f}\n'
        if not np.isnan(peak_corr):
            stats_text += f'Peak Corr: {peak_corr:.3f}\n'
        elif np.sum(peak_mask) == 1:
            stats_text += f'Peak Corr: N/A (1 pt)\n'
        else:
            stats_text += f'Peak Corr: N/A (no peaks)\n'
        stats_text += f'Amplitude: {amplitude:.2f}\n'
        stats_text += f'Num Peaks: {num_peaks}\n'
        stats_text += f'MSE: {np.mean(error**2):.4f}\n'
        stats_text += f'MAE: {np.mean(error):.4f}'
        
        axes[plot_idx, 2].text(0.02, 0.98, stats_text, 
                      transform=axes[plot_idx, 2].transAxes, 
                      verticalalignment='top',
                      fontsize=8,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle(f'Session {session_id} - Single Neuron Reconstructions', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def evaluate_session(model, session_id, device, window_size, stride, max_samples=None):
    """
    Valuta il modello su una specifica sessione (1 neurone)
    """
    
    print(f"\n{'='*70}")
    print(f"📊 Testando sessione: {session_id}")
    print(f"{'='*70}")
    
    try:
        # Carica dataset con 1 NEURONE
        dataset = UnifiedMultiSessionDataset(
            session_ids=[session_id],
            window_size=window_size,
            stride=stride,
            min_neurons=1,  # ✅ 1 NEURONE!
            mode='train'    # ✅ TRAIN mode = random selection (best neuron)
        )
        
        print(f"   🔴 SINGLE NEURON MODE")
        print(f"   🔴 Testing best neuron from session")
        
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
                neural_data = batch_data.to(device)
                
                # Forward pass
                vq_loss, neural_recon, perplexity, _, _ = model(neural_data)
                
                # Converti in numpy
                original = neural_data.cpu().numpy()
                reconstructed = neural_recon.cpu().numpy()
                
                # 💾 SALVA primi 20 campioni per visualizzazione
                if batch_idx == 0:
                    all_originals = original[:20]
                    all_reconstructions = reconstructed[:20]
                
                # Calcola metriche per TUTTI i campioni del batch
                for i in range(original.shape[0]):
                    if max_samples and samples_processed >= max_samples:
                        break
                    
                    sample_orig = original[i, 0, :]  # Shape: (60,)
                    sample_recon = reconstructed[i, 0, :]
                    
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
        
        # 📤 LOG TRACCE SU W&B (invece di heatmap!)
        print(f"\n📤 Loggando tracce su W&B per sessione {session_id}...")
        log_trace_examples_wandb(
            all_originals, all_reconstructions, session_id, session_label
        )
        print(f"✅ Tracce loggate su W&B!")
        
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


def log_trace_examples_wandb(originals, reconstructions, session_id, session_label):
    """Crea e logga 4 grafici separati (uno per sample) su W&B"""
    
    num_samples = len(originals)
    
    # Calcola ampiezze
    amplitudes = []
    for i in range(num_samples):
        trace = originals[i, 0, :]
        amplitude = np.max(trace) - np.min(trace)
        amplitudes.append(amplitude)
    
    amplitudes = np.array(amplitudes)
    top_indices = np.argsort(amplitudes)[-4:][::-1]
    
    print(f"  📊 Visualizing TOP 4 samples by amplitude:")
    
    # ✅ LOOP: Un grafico per sample
    for rank, sample_idx in enumerate(top_indices, 1):
        print(f"     Rank {rank}: Sample {sample_idx}, Amplitude={amplitudes[sample_idx]:.3f}")
        
        # Crea figura per questo sample (solo 1 riga!)
        fig = visualize_single_neuron_traces(
            originals, 
            reconstructions, 
            session_id, 
            [sample_idx]  # ← Lista con 1 elemento
        )
        
        # Log con nome descrittivo
        wandb.log({
            f'{session_label}/reconstruction_rank{rank}_amp{amplitudes[sample_idx]:.2f}': wandb.Image(fig)
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
    # FIGURA: Confronto metriche tra tutte le sessioni test
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🎯 CROSS-SESSION TEST: Single Neuron Generalization', 
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
    colors = ['lightcoral'] * len(session_names)
    
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
    axes[0, 1].set_ylim([0, 1.0])
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
    
    stats_text = "SINGLE NEURON GENERALIZATION:\n\n"
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
    plt.savefig(save_dir / 'cross_session_single_neuron_comparison.png', 
                dpi=200, bbox_inches='tight')
    
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
    print("🎯 CROSS-SESSION TEST: Single Neuron Model")
    print("="*70)
    print(f"📁 Modello: {CHECKPOINT_PATH}")
    print(f"🔬 Sessioni test: {TEST_SESSION_IDS}")
    print(f"📊 Modalità: {'FULL DATASET' if USE_FULL_DATASET else f'LIMITED ({MAX_SAMPLES} samples)'}")
    print(f"📐 Parametri: 1 neurone, window={WINDOW_SIZE}, stride={STRIDE}")
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
        tags=["cross-session", "single-neuron", "generalization-test"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    # Carica modello
    model = load_trained_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
    
    # Dizionario per salvare risultati
    all_results = {}
    
    # Test su sessioni nuove
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
        if avg_corr > 0.90:
            generalization_quality = "Excellent"
            print(f"   ✅ Eccellente generalizzazione cross-session!")
        elif avg_corr > 0.80:
            generalization_quality = "Good"
            print(f"   ✅ Buona generalizzazione cross-session")
        elif avg_corr > 0.60:
            generalization_quality = "Moderate"
            print(f"   ⚠️  Generalizzazione moderata")
        else:
            generalization_quality = "Poor"
            print(f"   ❌ Generalizzazione scarsa")
        
        summary_metrics['generalization_quality'] = generalization_quality
    
    # ✅ LOG SUMMARY TABLE SU W&B (indentazione corretta!)
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