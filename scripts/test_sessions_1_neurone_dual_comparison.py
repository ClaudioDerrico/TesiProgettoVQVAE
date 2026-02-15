#!/usr/bin/env python3
"""
Script per testare e confrontare modelli SINGLE NEURON vs DUAL CODEBOOK su sessioni cross-session.

Confronta:
- Modello SINGLE: best_single_neuron_model_1024_pt2.pth
- Modello DUAL: best_dual_model_1024_constrained.pth

Visualizzazioni: TRACCE TEMPORALI con entrambe le ricostruzioni

Uso:
    python scripts/test_sessions_1_neurone_dual_comparison.py
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
from models.dual_vqvae import DualCalciumVQVAE
from datasets.calcium import TRAINING_SESSION_IDS, TEST_SESSION_IDS

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

WANDB_PROJECT = "calcium-vqvae-cross-session-single-vs-dual"
WANDB_RUN_NAME = f"test-single-vs-dual-1024-comparison"

# ✅ MODELLI DA CONFRONTARE
CHECKPOINT_PATH_SINGLE = "results/best_single_neuron_model_1024_pt2.pth"
CHECKPOINT_PATH_DUAL = "results/dual_codebook_constrained/best_dual_model_1024_constrained.pth"

SAVE_DIR = "cross_session_single_vs_dual_analysis"

# Configurazione modello SINGLE
MODEL_CONFIG_SINGLE = {
    'num_neurons': 1,           
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 1024,
    'embedding_dim': 128,
    'commitment_cost': 0.5,
    'dropout_rate': 0.0,
    'use_quantizer': True,
}

# Configurazione modello DUAL
MODEL_CONFIG_DUAL = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 1024,
    'embedding_dim': 128,
    'commitment_cost': 0.25,
    'dropout_rate': 0.1,
    'use_quantizer': True,
    'threshold': 0.0,
    'threshold_type': 'adaptive',
}

BATCH_SIZE = 64
WINDOW_SIZE = 60
STRIDE = 50

USE_FULL_DATASET = True
MAX_SAMPLES = None

# ============================================================================
# FUNZIONI HELPER
# ============================================================================

def load_trained_model_single(checkpoint_path, config, device):
    """Carica il modello SINGLE dai pesi salvati"""
    print(f"🔄 Caricando modello SINGLE da: {checkpoint_path}")
    
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
        print(f"✅ Pesi SINGLE caricati")
        
        if 'best_correlation' in checkpoint:
            print(f"📊 Best training correlation (SINGLE): {checkpoint['best_correlation']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Pesi SINGLE caricati")
    
    model = model.to(device)
    model.eval()
    
    return model


def load_trained_model_dual(checkpoint_path, config, device):
    """Carica il modello DUAL dai pesi salvati"""
    print(f"🔄 Caricando modello DUAL da: {checkpoint_path}")
    
    model = DualCalciumVQVAE(
        num_neurons=config['num_neurons'],
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        dropout_rate=config.get('dropout_rate', 0.0),
        use_quantizer=config.get('use_quantizer', True),
        threshold=config.get('threshold', 0.0),
        threshold_type=config.get('threshold_type', 'adaptive')
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Pesi DUAL caricati")
        
        if 'test_correlation' in checkpoint:
            print(f"📊 Best training correlation (DUAL): {checkpoint['test_correlation']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Pesi DUAL caricati")
    
    model = model.to(device)
    model.eval()
    
    return model


def visualize_single_neuron_traces_comparison(original, reconstructed_single, reconstructed_dual, 
                                               session_id, sample_indices):
    """
    Visualizza tracce del singolo neurone con CONFRONTO SINGLE vs DUAL
    
    Args:
        original: (n_samples, 1, 60)
        reconstructed_single: (n_samples, 1, 60)
        reconstructed_dual: (n_samples, 1, 60)
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
        recon_single_trace = reconstructed_single[sample_idx, 0, :]
        recon_dual_trace = reconstructed_dual[sample_idx, 0, :]
        
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
        
        # === Plot 1: Overlay con tutte e tre le tracce ===
        axes[plot_idx, 0].plot(time_points, orig_trace, 'b-', 
                               label='Original', alpha=0.8, linewidth=2.0)
        axes[plot_idx, 0].plot(time_points, recon_single_trace, 'r-', 
                               label='Reconstruction (Single)', alpha=0.7, linewidth=1.5)
        axes[plot_idx, 0].plot(time_points, recon_dual_trace, 'g-', 
                               label='Reconstruction (Dual)', alpha=0.7, linewidth=1.5)
        
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
        
        axes[plot_idx, 0].set_xlabel('Time')
        axes[plot_idx, 0].set_ylabel('Activity')
        axes[plot_idx, 0].set_title(f'Sample {sample_idx}: Peaks={num_peaks}, '
                                     f'Amp={amplitude:.2f}, Max={max_abs_value:.2f}')
        axes[plot_idx, 0].legend(loc='upper right', fontsize=8)
        axes[plot_idx, 0].grid(True, alpha=0.3)
        
        # === Plot 2: Error comparison (Single vs Dual) ===
        error_single = np.abs(orig_trace - recon_single_trace)
        error_dual = np.abs(orig_trace - recon_dual_trace)
        
        axes[plot_idx, 1].plot(time_points, error_single, 'r-', alpha=0.7, linewidth=1.5,
                               label='Error (Single)')
        axes[plot_idx, 1].plot(time_points, error_dual, 'g-', alpha=0.7, linewidth=1.5,
                               label='Error (Dual)')
        axes[plot_idx, 1].fill_between(time_points, 0, error_single, alpha=0.2, color='red')
        axes[plot_idx, 1].fill_between(time_points, 0, error_dual, alpha=0.2, color='green')
        
        # Evidenzia errore nelle zone dei picchi
        if num_peaks > 0:
            peak_error_single = error_single.copy()
            peak_error_single[~peak_mask] = 0
            peak_error_dual = error_dual.copy()
            peak_error_dual[~peak_mask] = 0
            
            mean_peak_err_single = np.mean(error_single[peak_mask]) if np.sum(peak_mask) > 0 else 0
            mean_peak_err_dual = np.mean(error_dual[peak_mask]) if np.sum(peak_mask) > 0 else 0
            
            axes[plot_idx, 1].set_title(f'Sample {sample_idx}: Error Comparison\n'
                                        f'Peak err: Single={mean_peak_err_single:.3f}, '
                                        f'Dual={mean_peak_err_dual:.3f}')
        else:
            axes[plot_idx, 1].set_title(f'Sample {sample_idx}: Error Comparison (No peaks)')
        
        axes[plot_idx, 1].set_xlabel('Time')
        axes[plot_idx, 1].set_ylabel('Absolute Error')
        axes[plot_idx, 1].legend(loc='upper right', fontsize=8)
        axes[plot_idx, 1].grid(True, alpha=0.3)
        
        # === Plot 3: Scatter plots comparison ===
        if np.std(orig_trace) > 1e-8 and np.std(recon_single_trace) > 1e-8:
            corr_single, _ = pearsonr(orig_trace, recon_single_trace)
            
            # Correlazione solo sui picchi (single)
            if np.sum(peak_mask) >= 2:
                peak_corr_single, _ = pearsonr(orig_trace[peak_mask], recon_single_trace[peak_mask])
            else:
                peak_corr_single = np.nan
        else:
            corr_single = 0
            peak_corr_single = np.nan
        
        if np.std(orig_trace) > 1e-8 and np.std(recon_dual_trace) > 1e-8:
            corr_dual, _ = pearsonr(orig_trace, recon_dual_trace)
            
            # Correlazione solo sui picchi (dual)
            if np.sum(peak_mask) >= 2:
                peak_corr_dual, _ = pearsonr(orig_trace[peak_mask], recon_dual_trace[peak_mask])
            else:
                peak_corr_dual = np.nan
        else:
            corr_dual = 0
            peak_corr_dual = np.nan
        
        # Scatter plot: Original vs Reconstructed (Single)
        axes[plot_idx, 2].scatter(orig_trace, recon_single_trace, alpha=0.5, s=20, 
                                 color='red', label=f'Single (r={corr_single:.3f})')
        
        # Scatter plot: Original vs Reconstructed (Dual)
        axes[plot_idx, 2].scatter(orig_trace, recon_dual_trace, alpha=0.5, s=20, 
                                 color='green', label=f'Dual (r={corr_dual:.3f})')
        
        # Linea identità
        min_val = min(np.min(orig_trace), np.min(recon_single_trace), np.min(recon_dual_trace))
        max_val = max(np.max(orig_trace), np.max(recon_single_trace), np.max(recon_dual_trace))
        axes[plot_idx, 2].plot([min_val, max_val], [min_val, max_val], 
                               'k--', alpha=0.5, linewidth=1, label='Perfect reconstruction')
        
        axes[plot_idx, 2].set_xlabel('Original Activity')
        axes[plot_idx, 2].set_ylabel('Reconstructed Activity')
        axes[plot_idx, 2].set_title(f'Sample {sample_idx}: Correlation Comparison')
        axes[plot_idx, 2].legend(loc='upper left', fontsize=8)
        axes[plot_idx, 2].grid(True, alpha=0.3)
        
        # Box con statistiche
        mse_single = np.mean((orig_trace - recon_single_trace) ** 2)
        mse_dual = np.mean((orig_trace - recon_dual_trace) ** 2)
        mae_single = np.mean(error_single)
        mae_dual = np.mean(error_dual)
        
        stats_text = f'SINGLE MODEL:\n'
        stats_text += f'  Corr: {corr_single:.3f}\n'
        if not np.isnan(peak_corr_single):
            stats_text += f'  Peak Corr: {peak_corr_single:.3f}\n'
        stats_text += f'  MSE: {mse_single:.4f}\n'
        stats_text += f'  MAE: {mae_single:.4f}\n\n'
        
        stats_text += f'DUAL MODEL:\n'
        stats_text += f'  Corr: {corr_dual:.3f}\n'
        if not np.isnan(peak_corr_dual):
            stats_text += f'  Peak Corr: {peak_corr_dual:.3f}\n'
        stats_text += f'  MSE: {mse_dual:.4f}\n'
        stats_text += f'  MAE: {mae_dual:.4f}\n\n'
        
        stats_text += f'Amplitude: {amplitude:.2f}\n'
        stats_text += f'Num Peaks: {num_peaks}'
        
        axes[plot_idx, 2].text(0.02, 0.98, stats_text, 
                      transform=axes[plot_idx, 2].transAxes, 
                      verticalalignment='top',
                      fontsize=8,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle(f'Session {session_id} - Single vs Dual Model Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def evaluate_session_comparison(model_single, model_dual, session_id, device, 
                                window_size, stride, max_samples=None):
    """
    Valuta entrambi i modelli su una specifica sessione (1 neurone)
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
        
        # Metriche per SINGLE
        all_mse_single = []
        all_mae_single = []
        all_corr_single = []
        
        # Metriche per DUAL
        all_mse_dual = []
        all_mae_dual = []
        all_corr_dual = []
        
        all_originals = []
        all_reconstructions_single = []
        all_reconstructions_dual = []
        
        samples_processed = 0
        
        model_single.eval()
        model_dual.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                neural_data = batch_data.to(device)
                
                # Forward pass SINGLE
                vq_loss_single, neural_recon_single, perplexity_single, _, _ = model_single(neural_data)
                
                # Forward pass DUAL
                vq_loss_dual, neural_recon_dual, perplexity_dual, _, _ = model_dual(neural_data)
                
                # Converti in numpy
                original = neural_data.cpu().numpy()
                reconstructed_single = neural_recon_single.cpu().numpy()
                reconstructed_dual = neural_recon_dual.cpu().numpy()
                
                # 💾 SALVA primi 20 campioni per visualizzazione
                if batch_idx == 0:
                    all_originals = original[:20]
                    all_reconstructions_single = reconstructed_single[:20]
                    all_reconstructions_dual = reconstructed_dual[:20]
                
                # Calcola metriche per TUTTI i campioni del batch
                for i in range(original.shape[0]):
                    if max_samples and samples_processed >= max_samples:
                        break
                    
                    sample_orig = original[i, 0, :]  # Shape: (60,)
                    sample_recon_single = reconstructed_single[i, 0, :]
                    sample_recon_dual = reconstructed_dual[i, 0, :]
                    
                    # Metriche SINGLE
                    mse_single = np.mean((sample_orig - sample_recon_single) ** 2)
                    mae_single = np.mean(np.abs(sample_orig - sample_recon_single))
                    
                    if np.std(sample_orig) > 1e-8 and np.std(sample_recon_single) > 1e-8:
                        corr_single, _ = pearsonr(sample_orig, sample_recon_single)
                        corr_single = corr_single if np.isfinite(corr_single) else 0
                    else:
                        corr_single = 0
                    
                    # Metriche DUAL
                    mse_dual = np.mean((sample_orig - sample_recon_dual) ** 2)
                    mae_dual = np.mean(np.abs(sample_orig - sample_recon_dual))
                    
                    if np.std(sample_orig) > 1e-8 and np.std(sample_recon_dual) > 1e-8:
                        corr_dual, _ = pearsonr(sample_orig, sample_recon_dual)
                        corr_dual = corr_dual if np.isfinite(corr_dual) else 0
                    else:
                        corr_dual = 0
                    
                    all_mse_single.append(mse_single)
                    all_mae_single.append(mae_single)
                    all_corr_single.append(corr_single)
                    
                    all_mse_dual.append(mse_dual)
                    all_mae_dual.append(mae_dual)
                    all_corr_dual.append(corr_dual)
                    
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
            'num_samples': len(all_mse_single),
            'total_dataset_size': total_samples,
            
            # SINGLE metrics
            'mse_mean_single': np.mean(all_mse_single),
            'mse_std_single': np.std(all_mse_single),
            'mae_mean_single': np.mean(all_mae_single),
            'corr_mean_single': np.mean(all_corr_single),
            'corr_std_single': np.std(all_corr_single),
            
            # DUAL metrics
            'mse_mean_dual': np.mean(all_mse_dual),
            'mse_std_dual': np.std(all_mse_dual),
            'mae_mean_dual': np.mean(all_mae_dual),
            'corr_mean_dual': np.mean(all_corr_dual),
            'corr_std_dual': np.std(all_corr_dual),
            
            # Raw data
            'all_mse_single': all_mse_single,
            'all_corr_single': all_corr_single,
            'all_mse_dual': all_mse_dual,
            'all_corr_dual': all_corr_dual,
            
            # Visualizations
            'originals': all_originals,
            'reconstructions_single': all_reconstructions_single,
            'reconstructions_dual': all_reconstructions_dual,
            'success': True
        }
        
        print(f"\n📈 RISULTATI:")
        print(f"   Campioni analizzati: {results['num_samples']}/{total_samples}")
        print(f"\n   🔴 SINGLE MODEL:")
        print(f"      MSE:    {results['mse_mean_single']:.6f} ± {results['mse_std_single']:.6f}")
        print(f"      Corr:   {results['corr_mean_single']:.4f} ± {results['corr_std_single']:.4f}")
        print(f"\n   🟢 DUAL MODEL:")
        print(f"      MSE:    {results['mse_mean_dual']:.6f} ± {results['mse_std_dual']:.6f}")
        print(f"      Corr:   {results['corr_mean_dual']:.4f} ± {results['corr_std_dual']:.4f}")
        
        # 📤 LOG SU W&B - Metriche
        session_label = f"test_{session_id}"
        
        wandb.log({
            f'{session_label}/mse_mean_single': results['mse_mean_single'],
            f'{session_label}/corr_mean_single': results['corr_mean_single'],
            f'{session_label}/mse_mean_dual': results['mse_mean_dual'],
            f'{session_label}/corr_mean_dual': results['corr_mean_dual'],
            f'{session_label}/num_samples': results['num_samples'],
        })
        
        wandb.log({
            f'{session_label}/mse_distribution_single': wandb.Histogram(all_mse_single),
            f'{session_label}/correlation_distribution_single': wandb.Histogram(all_corr_single),
            f'{session_label}/mse_distribution_dual': wandb.Histogram(all_mse_dual),
            f'{session_label}/correlation_distribution_dual': wandb.Histogram(all_corr_dual),
        })
        
        # 📤 LOG TRACCE SU W&B
        print(f"\n📤 Loggando tracce su W&B per sessione {session_id}...")
        log_trace_examples_wandb(
            all_originals, all_reconstructions_single, all_reconstructions_dual, 
            session_id, session_label
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


def log_trace_examples_wandb(originals, reconstructions_single, reconstructions_dual, 
                             session_id, session_label):
    """Crea e logga 4 grafici separati (uno per sample) su W&B con confronto SINGLE vs DUAL"""
    
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
        fig = visualize_single_neuron_traces_comparison(
            originals, 
            reconstructions_single,
            reconstructions_dual,
            session_id, 
            [sample_idx]  # ← Lista con 1 elemento
        )
        
        # Log con nome descrittivo
        wandb.log({
            f'{session_label}/comparison_rank{rank}_amp{amplitudes[sample_idx]:.2f}': wandb.Image(fig)
        })
        
        plt.close(fig)


def visualize_cross_session_results_comparison(results_dict, save_dir):
    """Crea visualizzazioni comparative tra sessioni con confronto SINGLE vs DUAL"""
    
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎯 CROSS-SESSION TEST: Single vs Dual Model Comparison', 
                 fontsize=16, fontweight='bold')
    
    session_names = []
    mse_means_single = []
    mse_stds_single = []
    corr_means_single = []
    corr_stds_single = []
    mse_means_dual = []
    mse_stds_dual = []
    corr_means_dual = []
    corr_stds_dual = []
    
    for session_id, results in successful_results.items():
        label = f"Test {session_id}"
        session_names.append(label)
        mse_means_single.append(results['mse_mean_single'])
        mse_stds_single.append(results['mse_std_single'])
        corr_means_single.append(results['corr_mean_single'])
        corr_stds_single.append(results['corr_std_single'])
        mse_means_dual.append(results['mse_mean_dual'])
        mse_stds_dual.append(results['mse_std_dual'])
        corr_means_dual.append(results['corr_mean_dual'])
        corr_stds_dual.append(results['corr_std_dual'])
    
    # Bar plot MSE - Comparison
    x_pos = np.arange(len(session_names))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, mse_means_single, width, yerr=mse_stds_single, 
                   color='red', alpha=0.7, capsize=5, label='Single')
    axes[0, 0].bar(x_pos + width/2, mse_means_dual, width, yerr=mse_stds_dual, 
                   color='green', alpha=0.7, capsize=5, label='Dual')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(session_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Mean Squared Error: Single vs Dual')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Bar plot Correlation - Comparison
    axes[0, 1].bar(x_pos - width/2, corr_means_single, width, yerr=corr_stds_single, 
                   color='red', alpha=0.7, capsize=5, label='Single')
    axes[0, 1].bar(x_pos + width/2, corr_means_dual, width, yerr=corr_stds_dual, 
                   color='green', alpha=0.7, capsize=5, label='Dual')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(session_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_title('Mean Correlation: Single vs Dual')
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Distribution MSE - Comparison
    mse_distributions_single = [results['all_mse_single'] for results in successful_results.values()]
    mse_distributions_dual = [results['all_mse_dual'] for results in successful_results.values()]
    
    # Combine for boxplot
    combined_mse = []
    labels_combined = []
    for i, (single_dist, dual_dist) in enumerate(zip(mse_distributions_single, mse_distributions_dual)):
        combined_mse.append(single_dist)
        combined_mse.append(dual_dist)
        labels_combined.append(f"{session_names[i]}\nSingle")
        labels_combined.append(f"{session_names[i]}\nDual")
    
    bp1 = axes[1, 0].boxplot(combined_mse, labels=labels_combined, patch_artist=True)
    colors_box = ['red', 'green'] * len(session_names)
    for patch, color in zip(bp1['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_xticklabels(labels_combined, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('MSE Distribution: Single vs Dual')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Distribution Correlation - Comparison
    corr_distributions_single = [results['all_corr_single'] for results in successful_results.values()]
    corr_distributions_dual = [results['all_corr_dual'] for results in successful_results.values()]
    
    combined_corr = []
    for single_dist, dual_dist in zip(corr_distributions_single, corr_distributions_dual):
        combined_corr.append(single_dist)
        combined_corr.append(dual_dist)
    
    bp2 = axes[1, 1].boxplot(combined_corr, labels=labels_combined, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 1].set_xticklabels(labels_combined, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].set_title('Correlation Distribution: Single vs Dual')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Aggiungi statistiche
    avg_samples = int(np.mean([v['num_samples'] for v in successful_results.values()]))
    avg_mse_single = np.mean([v['mse_mean_single'] for v in successful_results.values()])
    avg_corr_single = np.mean([v['corr_mean_single'] for v in successful_results.values()])
    avg_mse_dual = np.mean([v['mse_mean_dual'] for v in successful_results.values()])
    avg_corr_dual = np.mean([v['corr_mean_dual'] for v in successful_results.values()])
    
    stats_text = "SINGLE vs DUAL COMPARISON:\n\n"
    stats_text += f"Test Sessions: {len(successful_results)}\n"
    stats_text += f"Avg Samples per Session: {avg_samples}\n\n"
    stats_text += f"SINGLE MODEL (Average):\n"
    stats_text += f"  MSE:  {avg_mse_single:.6f}\n"
    stats_text += f"  Corr: {avg_corr_single:.4f}\n\n"
    stats_text += f"DUAL MODEL (Average):\n"
    stats_text += f"  MSE:  {avg_mse_dual:.6f}\n"
    stats_text += f"  Corr: {avg_corr_dual:.4f}\n\n"
    stats_text += f"IMPROVEMENT:\n"
    stats_text += f"  MSE:  {((avg_mse_single - avg_mse_dual) / avg_mse_single * 100):.1f}%\n"
    stats_text += f"  Corr: {((avg_corr_dual - avg_corr_single) / avg_corr_single * 100):.1f}%"
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Salva localmente
    plt.savefig(save_dir / 'cross_session_single_vs_dual_comparison.png', 
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
    print("🎯 CROSS-SESSION TEST: Single vs Dual Model Comparison")
    print("="*70)
    print(f"📁 Modello SINGLE: {CHECKPOINT_PATH_SINGLE}")
    print(f"📁 Modello DUAL: {CHECKPOINT_PATH_DUAL}")
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
            "checkpoint_path_single": CHECKPOINT_PATH_SINGLE,
            "checkpoint_path_dual": CHECKPOINT_PATH_DUAL,
            "test_sessions": TEST_SESSION_IDS,
            "num_training_sessions": len(TRAINING_SESSION_IDS),
            "batch_size": BATCH_SIZE,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "use_full_dataset": USE_FULL_DATASET,
            "max_samples": MAX_SAMPLES,
            "model_config_single": MODEL_CONFIG_SINGLE,
            "model_config_dual": MODEL_CONFIG_DUAL,
        },
        tags=["cross-session", "single-neuron", "single-vs-dual", "comparison"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    # Carica entrambi i modelli
    print("\n" + "="*70)
    print("🔄 Caricamento modelli...")
    print("="*70)
    model_single = load_trained_model_single(CHECKPOINT_PATH_SINGLE, MODEL_CONFIG_SINGLE, device)
    model_dual = load_trained_model_dual(CHECKPOINT_PATH_DUAL, MODEL_CONFIG_DUAL, device)
    
    # Dizionario per salvare risultati
    all_results = {}
    
    # Test su sessioni nuove
    print("\n" + "="*70)
    print("🔬 CROSS-SESSION TEST: Sessioni Mai Viste")
    print("="*70)
    
    max_samples = None if USE_FULL_DATASET else MAX_SAMPLES
    
    for session_id in TEST_SESSION_IDS:
        results = evaluate_session_comparison(
            model_single, model_dual, session_id, device, WINDOW_SIZE, STRIDE, max_samples
        )
        all_results[session_id] = results
    
    # Crea visualizzazioni comparative
    print("\n" + "="*70)
    print("🎨 Creando visualizzazioni comparative...")
    print("="*70)
    visualize_cross_session_results_comparison(all_results, save_dir)
    
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
        avg_mse_single = np.mean([r['mse_mean_single'] for r in test_session_results])
        avg_corr_single = np.mean([r['corr_mean_single'] for r in test_session_results])
        avg_mse_dual = np.mean([r['mse_mean_dual'] for r in test_session_results])
        avg_corr_dual = np.mean([r['corr_mean_dual'] for r in test_session_results])
        avg_samples = int(np.mean([r['num_samples'] for r in test_session_results]))
        
        print(f"\n🔬 TEST SESSIONS (Average over {len(test_session_results)} sessions):")
        print(f"   Samples: {avg_samples} per session")
        print(f"\n   🔴 SINGLE MODEL:")
        print(f"      MSE:  {avg_mse_single:.6f}")
        print(f"      Corr: {avg_corr_single:.4f}")
        print(f"\n   🟢 DUAL MODEL:")
        print(f"      MSE:  {avg_mse_dual:.6f}")
        print(f"      Corr: {avg_corr_dual:.4f}")
        print(f"\n   📈 IMPROVEMENT:")
        mse_improvement = ((avg_mse_single - avg_mse_dual) / avg_mse_single * 100) if avg_mse_single > 0 else 0
        corr_improvement = ((avg_corr_dual - avg_corr_single) / avg_corr_single * 100) if avg_corr_single > 0 else 0
        print(f"      MSE:  {mse_improvement:+.1f}%")
        print(f"      Corr: {corr_improvement:+.1f}%")
        
        summary_metrics['test_sessions_mse_avg_single'] = avg_mse_single
        summary_metrics['test_sessions_corr_avg_single'] = avg_corr_single
        summary_metrics['test_sessions_mse_avg_dual'] = avg_mse_dual
        summary_metrics['test_sessions_corr_avg_dual'] = avg_corr_dual
        summary_metrics['test_sessions_samples_avg'] = avg_samples
        summary_metrics['num_test_sessions'] = len(test_session_results)
        summary_metrics['mse_improvement_pct'] = mse_improvement
        summary_metrics['corr_improvement_pct'] = corr_improvement
    
    # ✅ LOG SUMMARY TABLE SU W&B
    summary_table_data = []
    for session_id, results in all_results.items():
        if results['success']:
            summary_table_data.append([
                session_id,
                results['num_samples'],
                results['total_dataset_size'],
                f"{100*results['num_samples']/results['total_dataset_size']:.1f}%",
                results['mse_mean_single'],
                results['mse_std_single'],
                results['corr_mean_single'],
                results['corr_std_single'],
                results['mse_mean_dual'],
                results['mse_std_dual'],
                results['corr_mean_dual'],
                results['corr_std_dual'],
            ])
    
    summary_table = wandb.Table(
        columns=["Session ID", "Samples", "Total", "Coverage %", 
                 "MSE Single", "MSE Std Single", "Corr Single", "Corr Std Single",
                 "MSE Dual", "MSE Std Dual", "Corr Dual", "Corr Std Dual"],
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

