#!/usr/bin/env python3
"""
Analisi Correlazione vs Distanza dal Codice Base 0 (WANDB ONLY VERSION)

✨ MODIFICHE:
- Rimossi TUTTI i salvataggi locali (save_dir, savefig)
- SOLO upload su W&B
- Figure create in memoria e caricate direttamente su wandb
- Più pulito e cloud-first
- 4 GRAFICI: Temporal vs Cosine, Spectral vs Cosine, Temporal vs Euclidean, Spectral vs Euclidean

Uso:
    python codebook_correlation_vs_distance_WANDB_ONLY.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr, spearmanr, kendalltau, sem
from io import BytesIO
from PIL import Image

from models.vqvae import CalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/best_single_neuron_model_32.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-correlation-vs-distance"
WANDB_RUN_NAME = f"single_neuron_model_32"

MODEL_CONFIG = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 32,
    'embedding_dim': 32,
    'commitment_cost': 0.25,
    'dropout_rate': 0.1,
    'use_quantizer': True,
}

# Parametri
TEMPORAL_LENGTH = 15
BASE_CODE = 1
BATCH_SIZE = 8

# ============================================================================
# FUNZIONI - CARICAMENTO MODELLO
# ============================================================================

def load_model(checkpoint_path, config, device):
    """Carica il modello allenato"""
    print(f"🔄 Caricando modello da: {checkpoint_path}")
    
    model = CalciumVQVAE(
        num_neurons=config['num_neurons'],
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        dropout_rate=config['dropout_rate'],
        use_quantizer=config['use_quantizer']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Modello caricato (epoca {checkpoint.get('epoch', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Modello caricato")
    
    model = model.to(device)
    model.eval()
    
    return model


def extract_codebook_embeddings(model):
    """Estrae gli embeddings del codebook"""
    embeddings = model.vector_quantization.embedding.weight.data.cpu().numpy()
    print(f"✅ Codebook embeddings estratti: {embeddings.shape}")
    return embeddings

# ============================================================================
# FUNZIONI - DECODIFICA
# ============================================================================

def decode_code(model, code_idx, device, temporal_length):
    """Decodifica un singolo codice"""
    code_embedding = model.vector_quantization.embedding.weight[code_idx]
    sequence = code_embedding.unsqueeze(1).repeat(1, temporal_length)
    sequence = sequence.unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstruction = model.decode(sequence)
    
    trace = reconstruction.cpu().numpy()[0, 0, :]
    return trace


def decode_codes_batch(model, code_indices, device, temporal_length, batch_size=8):
    """Decodifica multipli codici in batch"""
    all_traces = []
    num_codes = len(code_indices)
    
    model.eval()
    with torch.no_grad():
        for i in range(0, num_codes, batch_size):
            batch_indices = code_indices[i:i+batch_size]
            batch_embeddings = model.vector_quantization.embedding.weight[batch_indices]
            batch_sequences = batch_embeddings.unsqueeze(2).repeat(1, 1, temporal_length)
            batch_sequences = batch_sequences.to(device)
            
            reconstructions = model.decode(batch_sequences)
            all_traces.append(reconstructions.cpu().numpy()[:, 0, :])
    
    return np.concatenate(all_traces, axis=0)

# ============================================================================
# FUNZIONI - ANALISI SPETTRALE
# ============================================================================

def compute_fft_spectrum(trace, sample_rate=1.0):
    """Calcola lo spettro di frequenza usando FFT"""
    n = len(trace)
    trace_centered = trace - np.mean(trace)
    fft_vals = np.fft.fft(trace_centered)
    freqs = np.fft.fftfreq(n, d=1.0/sample_rate)
    
    positive_freq_idx = freqs >= 0
    freqs_positive = freqs[positive_freq_idx]
    fft_positive = fft_vals[positive_freq_idx]
    power = np.abs(fft_positive) ** 2
    
    return freqs_positive, power

# ============================================================================
# FUNZIONI - CORRELAZIONI
# ============================================================================

def compute_temporal_correlation(trace_a, trace_b):
    """Calcola correlazione temporale con handling robusto"""
    std_a = np.std(trace_a)
    std_b = np.std(trace_b)
    
    if std_a < 1e-8 or std_b < 1e-8:
        if np.allclose(trace_a, trace_b, atol=1e-8):
            return 1.0
        elif std_a < 1e-8 and std_b < 1e-8:
            return 0.0
        else:
            return np.nan
    
    try:
        corr, _ = pearsonr(trace_a, trace_b)
        return corr if not np.isnan(corr) else np.nan
    except:
        return np.nan


def compute_spectral_correlation(trace_a, trace_b):
    """Calcola correlazione spettrale con handling robusto"""
    freqs_a, power_a = compute_fft_spectrum(trace_a)
    freqs_b, power_b = compute_fft_spectrum(trace_b)
    
    power_a_norm = power_a / (np.sum(power_a) + 1e-8)
    power_b_norm = power_b / (np.sum(power_b) + 1e-8)
    
    if len(power_a_norm) > 1 and np.std(power_a_norm) > 1e-8 and np.std(power_b_norm) > 1e-8:
        try:
            corr, _ = pearsonr(power_a_norm, power_b_norm)
            return corr if not np.isnan(corr) else np.nan
        except:
            return np.nan
    return np.nan

# ============================================================================
# FUNZIONI - TEST STATISTICI
# ============================================================================

def compute_correlation_metrics(x, y):
    """Calcola multiple correlation metrics"""
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    if len(x_valid) < 3:
        return {
            'pearson_r': np.nan, 'pearson_p': np.nan,
            'spearman_r': np.nan, 'spearman_p': np.nan,
            'kendall_tau': np.nan, 'kendall_p': np.nan,
            'n_valid': len(x_valid)
        }
    
    try:
        pearson_r, pearson_p = pearsonr(x_valid, y_valid)
    except:
        pearson_r, pearson_p = np.nan, np.nan
    
    try:
        spearman_r, spearman_p = spearmanr(x_valid, y_valid)
    except:
        spearman_r, spearman_p = np.nan, np.nan
    
    try:
        kendall_tau, kendall_p = kendalltau(x_valid, y_valid)
    except:
        kendall_tau, kendall_p = np.nan, np.nan
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p,
        'n_valid': len(x_valid)
    }

# ============================================================================
# FUNZIONI - ANALISI PRINCIPALE
# ============================================================================

def analyze_code0_correlations_vs_distance(model, embeddings, device, temporal_length, batch_size=8):
    """Analizza correlazioni vs distanza con batch decoding"""
    
    num_codes = embeddings.shape[0]
    base_code_idx = 1
    
    print(f"   🔵 Decodificando codice base 1...")
    base_trace = decode_code(model, base_code_idx, device, temporal_length)
    
    base_embedding = embeddings[base_code_idx:base_code_idx+1]
    similarity_to_base = cosine_similarity(embeddings, base_embedding).flatten()
    distances_to_base = 1 - similarity_to_base
    
    code_indices = np.arange(num_codes)
    mask = code_indices != base_code_idx
    
    other_codes = code_indices[mask]
    other_distances = distances_to_base[mask]
    
    sorted_indices = np.argsort(other_distances)
    sorted_codes = other_codes[sorted_indices]
    sorted_distances = other_distances[sorted_indices]
    
    print(f"   📊 Decodificando {len(sorted_codes)} codici in batch (batch_size={batch_size})...")
    all_traces = decode_codes_batch(model, sorted_codes, device, temporal_length, batch_size=batch_size)
    
    temporal_correlations = []
    spectral_correlations = []
    
    print(f"   📊 Calcolando correlazioni per {len(sorted_codes)} codici...")
    
    for idx, trace in enumerate(all_traces):
        if (idx + 1) % 10 == 0:
            print(f"      Processati {idx + 1}/{len(sorted_codes)} codici...")
        
        temp_corr = compute_temporal_correlation(base_trace, trace)
        temporal_correlations.append(temp_corr)
        
        spec_corr = compute_spectral_correlation(base_trace, trace)
        spectral_correlations.append(spec_corr)
    
    print(f"   ✅ Correlazioni calcolate!")
    
    return {
        'base_code': base_code_idx,
        'code_indices': np.concatenate([[base_code_idx], sorted_codes]),
        'distances': np.concatenate([[0.0], sorted_distances]),
        'temporal_correlations': np.concatenate([[1.0], np.array(temporal_correlations)]),
        'spectral_correlations': np.concatenate([[1.0], np.array(spectral_correlations)]),
        'base_trace': base_trace
    }

# ============================================================================
# FUNZIONI - STATISTICHE DESCRITTIVE
# ============================================================================

def print_correlation_statistics(results):
    """Stampa statistiche descrittive dettagliate"""
    temp_corr = results['temporal_correlations']
    spec_corr = results['spectral_correlations']
    distances = results['distances']
    
    temp_valid = temp_corr[~np.isnan(temp_corr)]
    spec_valid = spec_corr[~np.isnan(spec_corr)]
    
    print(f"\n{'='*70}")
    print("📊 STATISTICHE CORRELAZIONI DETTAGLIATE")
    print(f"{'='*70}")
    
    print(f"\n🔵 TEMPORAL:")
    print(f"   Valid samples: {len(temp_valid)}/{len(temp_corr)}")
    print(f"   High similarity (>0.9): {np.sum(temp_valid > 0.9)} codici ({100*np.sum(temp_valid > 0.9)/len(temp_valid):.1f}%)")
    print(f"   Medium similarity (0.5-0.9): {np.sum((temp_valid >= 0.5) & (temp_valid <= 0.9))} codici")
    print(f"   Low similarity (0-0.5): {np.sum((temp_valid >= 0) & (temp_valid < 0.5))} codici")
    print(f"   Uncorrelated (~0): {np.sum((temp_valid >= -0.1) & (temp_valid <= 0.1))} codici")
    print(f"   Anticorrelated (<0): {np.sum(temp_valid < 0)} codici ({100*np.sum(temp_valid < 0)/len(temp_valid):.1f}%)")
    print(f"   Strong anticorrelation (<-0.5): {np.sum(temp_valid < -0.5)} codici")
    
    print(f"\n🟣 SPECTRAL:")
    print(f"   Valid samples: {len(spec_valid)}/{len(spec_corr)}")
    print(f"   Near-identical (>0.95): {np.sum(spec_valid > 0.95)} codici ({100*np.sum(spec_valid > 0.95)/len(spec_valid):.1f}%)")
    print(f"   High similarity (0.8-0.95): {np.sum((spec_valid >= 0.8) & (spec_valid <= 0.95))} codici")
    print(f"   Medium similarity (0.5-0.8): {np.sum((spec_valid >= 0.5) & (spec_valid < 0.8))} codici")
    print(f"   Low similarity (<0.5): {np.sum(spec_valid < 0.5)} codici ({100*np.sum(spec_valid < 0.5)/len(spec_valid):.1f}%)")
    
    print(f"\n🟢 DISTANCES:")
    print(f"   Very close (<0.3): {np.sum(distances < 0.3)} codici")
    print(f"   Close (0.3-0.6): {np.sum((distances >= 0.3) & (distances < 0.6))} codici")
    print(f"   Far (0.6-1.0): {np.sum((distances >= 0.6) & (distances < 1.0))} codici")
    print(f"   Very far (>1.0): {np.sum(distances >= 1.0)} codici")
    
    print(f"\n⚠️  PARADOSSI (Disorganizzazione):")
    
    valid_mask = ~(np.isnan(temp_corr) | np.isnan(spec_corr))
    distances_valid = distances[valid_mask]
    temp_valid_full = temp_corr[valid_mask]
    spec_valid_full = spec_corr[valid_mask]
    
    paradox_1 = np.sum((distances_valid < 0.3) & (temp_valid_full < 0))
    print(f"   Close embeddings, negative temporal corr: {paradox_1} codici")
    
    paradox_2 = np.sum((distances_valid > 0.8) & (temp_valid_full > 0.8))
    print(f"   Far embeddings, high temporal corr: {paradox_2} codici")
    
    paradox_3 = np.sum((spec_valid_full > 0.95) & (temp_valid_full < 0.3))
    print(f"   High spectral, low temporal corr: {paradox_3} codici (PHASE ISSUE!)")
    
    paradox_4 = np.sum((temp_valid_full > 0.9) & (temp_valid_full != 1.0))
    print(f"   Near-duplicate codes (corr>0.9, not base): {paradox_4} codici (MODE COLLAPSE?)")

# ============================================================================
# HELPER: Converti matplotlib figure a PIL Image per W&B
# ============================================================================

def fig_to_pil(fig):
    """Converte matplotlib figure a PIL Image (per W&B)"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img

# ============================================================================
# FUNZIONI - VISUALIZZAZIONI (✨ WANDB ONLY)
# ============================================================================

def visualize_code0_correlation_vs_distance(results, embeddings):
    """
    ✨ WANDB ONLY: Crea SOLO i 4 grafici richiesti
    
    1. Temporal Correlation vs Cosine Distance
    2. Spectral Correlation vs Cosine Distance  
    3. Temporal Correlation vs Euclidean Distance
    4. Spectral Correlation vs Euclidean Distance
    """
    
    distances = results['distances']
    temp_corr = results['temporal_correlations']
    spec_corr = results['spectral_correlations']
    code_indices = results['code_indices']
    
    valid_mask = ~(np.isnan(temp_corr) | np.isnan(spec_corr))
    distances_valid = distances[valid_mask]
    temp_corr_valid = temp_corr[valid_mask]
    spec_corr_valid = spec_corr[valid_mask]
    code_indices_valid = code_indices[valid_mask]
    
    # Calcola distanze Euclidean
    base_code_idx = 1
    base_embedding = embeddings[base_code_idx:base_code_idx+1]
    euclidean_dists = euclidean_distances(embeddings, base_embedding).flatten()
    euclidean_dists_valid = euclidean_dists[valid_mask]
    
    wandb_images = {}
    
    # ========================================================================
    # UNICO PLOT: 2x2 con i 4 grafici richiesti
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # ========================================================================
    # PLOT 1 (Top-Left): Temporal Correlation vs Cosine Distance
    # ========================================================================
    ax = axes[0, 0]
    
    scatter1 = ax.scatter(distances_valid, temp_corr_valid, alpha=0.6, s=80, 
                         c=code_indices_valid, cmap='viridis', edgecolors='black', linewidth=0.8)
    
    ax.plot(distances_valid, temp_corr_valid, 'gray', alpha=0.15, linewidth=0.5, 
           label='Connection', zorder=1)
    
    # Polynomial fit (grado 2)
    z_poly = np.polyfit(distances_valid, temp_corr_valid, 2)
    p_poly = np.poly1d(z_poly)
    x_smooth = np.linspace(distances_valid.min(), distances_valid.max(), 200)
    ax.plot(x_smooth, p_poly(x_smooth), 'r--', linewidth=3, 
           label=f'Poly fit (deg 2)', zorder=3)
    
    # Base Code marker
    base_idx = np.where(code_indices_valid == base_code_idx)[0]
    if len(base_idx) > 0:
        ax.scatter(distances_valid[base_idx], temp_corr_valid[base_idx], 
                  marker='*', s=500, c='red', edgecolors='black', linewidth=2,
                  label=f'Base Code {base_code_idx}', zorder=5)
    
    ax.set_xlabel('Cosine Distance (Embedding Space)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temporal Correlation', fontsize=13, fontweight='bold')
    ax.set_title('Temporal Correlation vs Cosine Distance', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    
    cbar1 = plt.colorbar(scatter1, ax=ax, label='Code Index')
    
    # Stats box
    metrics_temp_cosine = compute_correlation_metrics(distances_valid, temp_corr_valid)
    stats_text = f'Pearson r = {metrics_temp_cosine["pearson_r"]:.4f}\n'
    stats_text += f'p = {metrics_temp_cosine["pearson_p"]:.2e}\n'
    stats_text += f'Mean corr = {np.mean(temp_corr_valid):.3f}\n'
    stats_text += f'Std = {np.std(temp_corr_valid):.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # ========================================================================
    # PLOT 2 (Top-Right): Spectral Correlation vs Cosine Distance
    # ========================================================================
    ax = axes[0, 1]
    
    scatter2 = ax.scatter(distances_valid, spec_corr_valid, alpha=0.6, s=80,
                         c=code_indices_valid, cmap='plasma', edgecolors='black', linewidth=0.8)
    
    ax.plot(distances_valid, spec_corr_valid, 'gray', alpha=0.15, linewidth=0.5,
           label='Connection', zorder=1)
    
    # Polynomial fit
    z_poly2 = np.polyfit(distances_valid, spec_corr_valid, 2)
    p_poly2 = np.poly1d(z_poly2)
    ax.plot(x_smooth, p_poly2(x_smooth), 'r--', linewidth=3,
           label=f'Poly fit (deg 2)', zorder=3)
    
    # Base Code marker
    if len(base_idx) > 0:
        ax.scatter(distances_valid[base_idx], spec_corr_valid[base_idx],
                  marker='*', s=500, c='red', edgecolors='black', linewidth=2,
                  label=f'Base Code {base_code_idx}', zorder=5)
    
    ax.set_xlabel('Cosine Distance (Embedding Space)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Spectral Correlation', fontsize=13, fontweight='bold')
    ax.set_title('Spectral Correlation vs Cosine Distance', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    
    cbar2 = plt.colorbar(scatter2, ax=ax, label='Code Index')
    
    # Stats box
    metrics_spec_cosine = compute_correlation_metrics(distances_valid, spec_corr_valid)
    stats_text = f'Pearson r = {metrics_spec_cosine["pearson_r"]:.4f}\n'
    stats_text += f'p = {metrics_spec_cosine["pearson_p"]:.2e}\n'
    stats_text += f'Mean corr = {np.mean(spec_corr_valid):.3f}\n'
    stats_text += f'Std = {np.std(spec_corr_valid):.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # ========================================================================
    # PLOT 3 (Bottom-Left): Temporal Correlation vs Euclidean Distance
    # ========================================================================
    ax = axes[1, 0]
    
    scatter3 = ax.scatter(euclidean_dists_valid, temp_corr_valid, alpha=0.6, s=80,
                         c=code_indices_valid, cmap='coolwarm', edgecolors='black', linewidth=0.8)
    
    ax.plot(euclidean_dists_valid, temp_corr_valid, 'gray', alpha=0.15, linewidth=0.5,
           label='Connection', zorder=1)
    
    # Polynomial fit
    z_poly3 = np.polyfit(euclidean_dists_valid, temp_corr_valid, 2)
    p_poly3 = np.poly1d(z_poly3)
    x_smooth3 = np.linspace(euclidean_dists_valid.min(), euclidean_dists_valid.max(), 200)
    ax.plot(x_smooth3, p_poly3(x_smooth3), 'r--', linewidth=3,
           label=f'Poly fit (deg 2)', zorder=3)
    
    # Base Code marker
    if len(base_idx) > 0:
        ax.scatter(euclidean_dists_valid[base_idx], temp_corr_valid[base_idx],
                  marker='*', s=500, c='red', edgecolors='black', linewidth=2,
                  label=f'Base Code {base_code_idx}', zorder=5)
    
    ax.set_xlabel('Euclidean Distance (Embedding Space)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temporal Correlation', fontsize=13, fontweight='bold')
    ax.set_title('Temporal Correlation vs Euclidean Distance', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    
    cbar3 = plt.colorbar(scatter3, ax=ax, label='Code Index')
    
    # Stats box
    metrics_temp_euclidean = compute_correlation_metrics(euclidean_dists_valid, temp_corr_valid)
    stats_text = f'Pearson r = {metrics_temp_euclidean["pearson_r"]:.4f}\n'
    stats_text += f'p = {metrics_temp_euclidean["pearson_p"]:.2e}\n'
    stats_text += f'Mean corr = {np.mean(temp_corr_valid):.3f}\n'
    stats_text += f'Std = {np.std(temp_corr_valid):.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # ========================================================================
    # PLOT 4 (Bottom-Right): Spectral Correlation vs Euclidean Distance
    # ========================================================================
    ax = axes[1, 1]
    
    scatter4 = ax.scatter(euclidean_dists_valid, spec_corr_valid, alpha=0.6, s=80,
                         c=code_indices_valid, cmap='RdYlGn', edgecolors='black', linewidth=0.8)
    
    ax.plot(euclidean_dists_valid, spec_corr_valid, 'gray', alpha=0.15, linewidth=0.5,
           label='Connection', zorder=1)
    
    # Polynomial fit
    z_poly4 = np.polyfit(euclidean_dists_valid, spec_corr_valid, 2)
    p_poly4 = np.poly1d(z_poly4)
    ax.plot(x_smooth3, p_poly4(x_smooth3), 'r--', linewidth=3,
           label=f'Poly fit (deg 2)', zorder=3)
    
    # Base Code marker
    if len(base_idx) > 0:
        ax.scatter(euclidean_dists_valid[base_idx], spec_corr_valid[base_idx],
                  marker='*', s=500, c='red', edgecolors='black', linewidth=2,
                  label=f'Base Code {base_code_idx}', zorder=5)
    
    ax.set_xlabel('Euclidean Distance (Embedding Space)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Spectral Correlation', fontsize=13, fontweight='bold')
    ax.set_title('Spectral Correlation vs Euclidean Distance', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    
    cbar4 = plt.colorbar(scatter4, ax=ax, label='Code Index')
    
    # Stats box
    metrics_spec_euclidean = compute_correlation_metrics(euclidean_dists_valid, spec_corr_valid)
    stats_text = f'Pearson r = {metrics_spec_euclidean["pearson_r"]:.4f}\n'
    stats_text += f'p = {metrics_spec_euclidean["pearson_p"]:.2e}\n'
    stats_text += f'Mean corr = {np.mean(spec_corr_valid):.3f}\n'
    stats_text += f'Std = {np.std(spec_corr_valid):.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # ========================================================================
    # Suptitle e finalizzazione
    # ========================================================================
    plt.suptitle(f'Correlation vs Distance from Base Code {base_code_idx}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Converti a PIL e aggiungi a dict
    wandb_images['correlation_vs_distance_4plots'] = wandb.Image(fig_to_pil(fig))
    plt.close(fig)
    
    print(f"✅ 4-plot figure creata per W&B")
    
    return wandb_images, {
        'metrics_temp_cosine': metrics_temp_cosine,
        'metrics_spec_cosine': metrics_spec_cosine,
        'metrics_temp_euclidean': metrics_temp_euclidean,
        'metrics_spec_euclidean': metrics_spec_euclidean,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("📊 ANALISI CORRELAZIONE vs DISTANZA - WANDB ONLY VERSION")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Temporal length: {TEMPORAL_LENGTH}")
    print(f"   Base code: {BASE_CODE}")
    print(f"   Batch size (decoding): {BATCH_SIZE}")
    print(f"   ✨ Mode: WANDB ONLY (no local saves)")
    print(f"   📊 4 Plots: Temp vs Cosine, Spec vs Cosine, Temp vs Euclidean, Spec vs Euclidean")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Inizializza W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "checkpoint_path": CHECKPOINT_PATH,
            "temporal_length": TEMPORAL_LENGTH,
            "base_code": BASE_CODE,
            "batch_size": BATCH_SIZE,
            **MODEL_CONFIG
        },
        tags=["correlation", "distance-analysis", "cosine-euclidean", "wandb-only"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # 1. Carica modello
        model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # 2. Estrai embeddings
        embeddings = extract_codebook_embeddings(model)
        num_codes = embeddings.shape[0]
        
        print(f"\n🔍 Totale codici nel codebook: {num_codes}")
        print(f"   Analizzando codice base: {BASE_CODE}")
        print(f"   Altri codici da confrontare: {num_codes - 1}")
        
        # 3. Analizza codice base
        print(f"\n{'='*70}")
        print(f"📊 ANALISI CODICE BASE {BASE_CODE}")
        print(f"{'='*70}")
        
        results = analyze_code0_correlations_vs_distance(
            model, embeddings, device, TEMPORAL_LENGTH, batch_size=BATCH_SIZE
        )
        
        # 4. Stampa statistiche
        print_correlation_statistics(results)
        
        # Log metriche base
        wandb.log({
            'mean_temporal_corr': np.nanmean(results['temporal_correlations']),
            'std_temporal_corr': np.nanstd(results['temporal_correlations']),
            'mean_spectral_corr': np.nanmean(results['spectral_correlations']),
            'std_spectral_corr': np.nanstd(results['spectral_correlations']),
            'mean_cosine_distance': np.mean(results['distances']),
            'std_cosine_distance': np.std(results['distances']),
        })
        
        # 5. Crea visualizzazioni (WANDB ONLY)
        print(f"\n{'='*70}")
        print("📈 CREAZIONE VISUALIZZAZIONI (W&B Upload)")
        print(f"{'='*70}")
        
        wandb_images, viz_stats = visualize_code0_correlation_vs_distance(results, embeddings)
        
        # 6. Upload immagini su W&B
        print(f"\n☁️  Uploading images to W&B...")
        wandb.log(wandb_images)
        print(f"✅ 1 immagine (4 sub-plots) caricata su W&B!")
        print(f"   - correlation_vs_distance_4plots:")
        print(f"     • Top-Left: Temporal Correlation vs Cosine Distance")
        print(f"     • Top-Right: Spectral Correlation vs Cosine Distance")
        print(f"     • Bottom-Left: Temporal Correlation vs Euclidean Distance")
        print(f"     • Bottom-Right: Spectral Correlation vs Euclidean Distance")
        
        # 7. Log statistiche finali
        wandb.log({
            # Cosine distance metrics
            'temp_cosine_pearson_r': viz_stats['metrics_temp_cosine']['pearson_r'],
            'temp_cosine_spearman_r': viz_stats['metrics_temp_cosine']['spearman_r'],
            'spec_cosine_pearson_r': viz_stats['metrics_spec_cosine']['pearson_r'],
            'spec_cosine_spearman_r': viz_stats['metrics_spec_cosine']['spearman_r'],
            # Euclidean distance metrics
            'temp_euclidean_pearson_r': viz_stats['metrics_temp_euclidean']['pearson_r'],
            'temp_euclidean_spearman_r': viz_stats['metrics_temp_euclidean']['spearman_r'],
            'spec_euclidean_pearson_r': viz_stats['metrics_spec_euclidean']['pearson_r'],
            'spec_euclidean_spearman_r': viz_stats['metrics_spec_euclidean']['spearman_r'],
        })
        
        # 8. Summary
        print(f"\n{'='*70}")
        print("📊 STATISTICHE FINALI")
        print(f"{'='*70}")
        
        print(f"\n   🔵 COSINE DISTANCE:")
        print(f"      Temporal - Pearson r: {viz_stats['metrics_temp_cosine']['pearson_r']:.4f}")
        print(f"      Spectral - Pearson r: {viz_stats['metrics_spec_cosine']['pearson_r']:.4f}")
        
        print(f"\n   🟢 EUCLIDEAN DISTANCE:")
        print(f"      Temporal - Pearson r: {viz_stats['metrics_temp_euclidean']['pearson_r']:.4f}")
        print(f"      Spectral - Pearson r: {viz_stats['metrics_spec_euclidean']['pearson_r']:.4f}")
        
        print(f"\n{'='*70}")
        print("✅ ANALISI COMPLETATA!")
        print(f"{'='*70}")
        print(f"\n🌐 W&B Dashboard: {wandb.run.url}")
        print(f"✨ Nessun file salvato in locale - tutto su W&B!")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        wandb.finish()
        print("\n✅ W&B run completato!")


if __name__ == "__main__":
    main()