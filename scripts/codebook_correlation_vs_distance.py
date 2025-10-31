#!/usr/bin/env python3
"""
Analisi Correlazione vs Distanza dal Codice Base 0

Ordina tutti gli altri codici per distanza crescente dal codice 0
e visualizza come temporal/spectral correlation decadono con la distanza.

Uso:
    python codebook_correlation_vs_distance_code0.py
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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from pathlib import Path

from models.vqvae import CalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/best_single_neuron_model_32.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-correlation-vs-distance"
WANDB_RUN_NAME = f"code0-corr-vs-dist-{datetime.now().strftime('%m%d-%H%M')}"

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
BASE_CODE = 0  # ✅ Analizziamo solo il codice 0
SAVE_DIR = Path("correlation_vs_distance_code0")

# ============================================================================
# FUNZIONI
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


def decode_code(model, code_idx, device, temporal_length):
    """
    Decodifica un singolo codice
    
    Returns:
        trace: (output_time,) - traccia temporale del neurone
    """
    code_embedding = model.vector_quantization.embedding.weight[code_idx]
    
    # Ripeti per temporal_length
    sequence = code_embedding.unsqueeze(1).repeat(1, temporal_length)
    sequence = sequence.unsqueeze(0).to(device)
    
    # Decodifica
    with torch.no_grad():
        reconstruction = model.decode(sequence)
    
    trace = reconstruction.cpu().numpy()[0, 0, :]  # (output_time,)
    
    return trace


def compute_fft_spectrum(trace, sample_rate=1.0):
    """
    Calcola lo spettro di frequenza usando FFT
    
    Returns:
        freqs: array delle frequenze (Hz)
        power: spettro di potenza (magnitudine al quadrato)
    """
    n = len(trace)
    
    # Rimuovi la componente DC (media)
    trace_centered = trace - np.mean(trace)
    
    # FFT
    fft_vals = np.fft.fft(trace_centered)
    
    # Frequenze corrispondenti (solo metà positiva)
    freqs = np.fft.fftfreq(n, d=1.0/sample_rate)
    
    # Solo frequenze positive
    positive_freq_idx = freqs >= 0
    freqs_positive = freqs[positive_freq_idx]
    fft_positive = fft_vals[positive_freq_idx]
    
    # Potenza
    power = np.abs(fft_positive) ** 2
    
    return freqs_positive, power


def compute_temporal_correlation(trace_a, trace_b):
    """Calcola la correlazione temporale tra due tracce"""
    if np.std(trace_a) > 1e-8 and np.std(trace_b) > 1e-8:
        corr, _ = pearsonr(trace_a, trace_b)
        if np.isnan(corr):
            return 0.0
        return corr
    return 0.0


def compute_spectral_correlation(trace_a, trace_b):
    """Calcola la correlazione spettrale tra due tracce"""
    # Calcola spettri
    freqs_a, power_a = compute_fft_spectrum(trace_a)
    freqs_b, power_b = compute_fft_spectrum(trace_b)
    
    # Normalizza power
    power_a_norm = power_a / (np.sum(power_a) + 1e-8)
    power_b_norm = power_b / (np.sum(power_b) + 1e-8)
    
    # Correlazione tra spettri
    if len(power_a_norm) > 1 and np.std(power_a_norm) > 1e-8 and np.std(power_b_norm) > 1e-8:
        corr, _ = pearsonr(power_a_norm, power_b_norm)
        if np.isnan(corr):
            return 0.0
        return corr
    return 0.0


def analyze_code0_correlations_vs_distance(model, embeddings, device, temporal_length):
    """
    Analizza come temporal e spectral correlation variano con la distanza
    dal codice 0.
    
    Returns:
        dict con:
            - distances: array delle distanze dal codice 0
            - temporal_correlations: correlazioni temporali
            - spectral_correlations: correlazioni spettrali
            - code_indices: indici dei codici ordinati per distanza
            - base_trace: traccia del codice 0
    """
    
    num_codes = embeddings.shape[0]
    base_code_idx = 0
    
    # Decodifica il codice 0
    print(f"   🔵 Decodificando codice base 0...")
    base_trace = decode_code(model, base_code_idx, device, temporal_length)
    
    # Calcola distanze dal codice 0
    base_embedding = embeddings[base_code_idx:base_code_idx+1]  # (1, embedding_dim)
    similarity_to_base = cosine_similarity(embeddings, base_embedding).flatten()
    distances_to_base = 1 - similarity_to_base
    
    # Ordina codici per distanza crescente (escludi il codice 0 stesso)
    code_indices = np.arange(num_codes)
    mask = code_indices != base_code_idx  # Escludi codice 0
    
    other_codes = code_indices[mask]
    other_distances = distances_to_base[mask]
    
    # Ordina per distanza crescente
    sorted_indices = np.argsort(other_distances)
    sorted_codes = other_codes[sorted_indices]
    sorted_distances = other_distances[sorted_indices]
    
    # Calcola correlazioni per ogni codice
    temporal_correlations = []
    spectral_correlations = []
    
    print(f"   📊 Calcolando correlazioni per {len(sorted_codes)} codici...")
    
    for idx, code_idx in enumerate(sorted_codes):
        # Progress ogni 10 codici
        if (idx + 1) % 10 == 0:
            print(f"      Processati {idx + 1}/{len(sorted_codes)} codici...")
        
        # Decodifica
        trace = decode_code(model, code_idx, device, temporal_length)
        
        # Temporal correlation
        temp_corr = compute_temporal_correlation(base_trace, trace)
        temporal_correlations.append(temp_corr)
        
        # Spectral correlation
        spec_corr = compute_spectral_correlation(base_trace, trace)
        spectral_correlations.append(spec_corr)
    
    print(f"   ✅ Correlazioni calcolate!")
    
    return {
        'base_code': base_code_idx,
        'code_indices': sorted_codes,
        'distances': sorted_distances,
        'temporal_correlations': np.array(temporal_correlations),
        'spectral_correlations': np.array(spectral_correlations),
        'base_trace': base_trace
    }


def visualize_code0_correlation_vs_distance(results, save_dir):
    """
    Visualizza come temporal/spectral correlation variano con la distanza
    dal codice 0.
    """
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    distances = results['distances']
    temp_corr = results['temporal_correlations']
    spec_corr = results['spectral_correlations']
    code_indices = results['code_indices']
    
    # ========================================================================
    # PLOT 1: Scatter plot con trend line
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- Temporal Correlation ---
    scatter1 = axes[0].scatter(distances, temp_corr, alpha=0.6, s=60, 
                              c=distances, cmap='viridis', edgecolors='black', linewidth=0.5)
    axes[0].plot(distances, temp_corr, 'r-', alpha=0.2, linewidth=1, label='Connection')
    
    # Trend line (regressione lineare)
    z_temp = np.polyfit(distances, temp_corr, 1)
    p_temp = np.poly1d(z_temp)
    axes[0].plot(distances, p_temp(distances), "b--", linewidth=3, 
                label=f'Linear fit: y={z_temp[0]:.3f}x+{z_temp[1]:.3f}')
    
    axes[0].set_xlabel('Distance from Code 0', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Temporal Correlation', fontsize=13, fontweight='bold')
    axes[0].set_title('Temporal Correlation vs Distance from Code 0', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    # Colorbar
    plt.colorbar(scatter1, ax=axes[0], label='Distance')
    
    # Statistiche
    corr_temp_dist, p_temp_val = pearsonr(distances, temp_corr)
    stats_text = f'Pearson r = {corr_temp_dist:.4f}\n'
    stats_text += f'p-value = {p_temp_val:.4e}\n'
    stats_text += f'Mean corr = {np.mean(temp_corr):.3f}\n'
    stats_text += f'Std = {np.std(temp_corr):.3f}\n'
    stats_text += f'Min = {np.min(temp_corr):.3f}\n'
    stats_text += f'Max = {np.max(temp_corr):.3f}'
    
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # --- Spectral Correlation ---
    scatter2 = axes[1].scatter(distances, spec_corr, alpha=0.6, s=60,
                              c=distances, cmap='plasma', edgecolors='black', linewidth=0.5)
    axes[1].plot(distances, spec_corr, 'b-', alpha=0.2, linewidth=1, label='Connection')
    
    # Trend line
    z_spec = np.polyfit(distances, spec_corr, 1)
    p_spec = np.poly1d(z_spec)
    axes[1].plot(distances, p_spec(distances), "orange", linestyle='--', linewidth=3,
                label=f'Linear fit: y={z_spec[0]:.3f}x+{z_spec[1]:.3f}')
    
    axes[1].set_xlabel('Distance from Code 0', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Spectral Correlation', fontsize=13, fontweight='bold')
    axes[1].set_title('Spectral Correlation vs Distance from Code 0',
                     fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    # Colorbar
    plt.colorbar(scatter2, ax=axes[1], label='Distance')
    
    # Statistiche
    corr_spec_dist, p_spec_val = pearsonr(distances, spec_corr)
    stats_text = f'Pearson r = {corr_spec_dist:.4f}\n'
    stats_text += f'p-value = {p_spec_val:.4e}\n'
    stats_text += f'Mean corr = {np.mean(spec_corr):.3f}\n'
    stats_text += f'Std = {np.std(spec_corr):.3f}\n'
    stats_text += f'Min = {np.min(spec_corr):.3f}\n'
    stats_text += f'Max = {np.max(spec_corr):.3f}'
    
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.suptitle(f'Correlation Decay Analysis - Base Code 0', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_dir / 'code0_correlation_vs_distance_main.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Main plot salvato: {save_dir / 'code0_correlation_vs_distance_main.png'}")
    
    # ========================================================================
    # PLOT 2: Binned analysis (distanza in bins)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Crea bins di distanza
    num_bins = 10
    bins = np.linspace(np.min(distances), np.max(distances), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    temp_corr_binned = []
    spec_corr_binned = []
    temp_corr_std = []
    spec_corr_std = []
    
    for i in range(num_bins):
        mask = (distances >= bins[i]) & (distances < bins[i+1])
        
        if np.sum(mask) > 0:
            temp_corr_binned.append(np.mean(temp_corr[mask]))
            spec_corr_binned.append(np.mean(spec_corr[mask]))
            temp_corr_std.append(np.std(temp_corr[mask]))
            spec_corr_std.append(np.std(spec_corr[mask]))
        else:
            temp_corr_binned.append(np.nan)
            spec_corr_binned.append(np.nan)
            temp_corr_std.append(np.nan)
            spec_corr_std.append(np.nan)
    
    temp_corr_binned = np.array(temp_corr_binned)
    spec_corr_binned = np.array(spec_corr_binned)
    temp_corr_std = np.array(temp_corr_std)
    spec_corr_std = np.array(spec_corr_std)
    
    # Temporal binned
    axes[0].errorbar(bin_centers, temp_corr_binned, yerr=temp_corr_std,
                    fmt='o-', color='blue', markersize=10, linewidth=2,
                    capsize=5, capthick=2, label='Mean ± Std')
    axes[0].fill_between(bin_centers, 
                        temp_corr_binned - temp_corr_std,
                        temp_corr_binned + temp_corr_std,
                        alpha=0.2, color='blue')
    
    axes[0].set_xlabel('Distance from Code 0 (binned)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Temporal Correlation', fontsize=13, fontweight='bold')
    axes[0].set_title('Binned Temporal Correlation', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
    
    # Spectral binned
    axes[1].errorbar(bin_centers, spec_corr_binned, yerr=spec_corr_std,
                    fmt='s-', color='red', markersize=10, linewidth=2,
                    capsize=5, capthick=2, label='Mean ± Std')
    axes[1].fill_between(bin_centers,
                        spec_corr_binned - spec_corr_std,
                        spec_corr_binned + spec_corr_std,
                        alpha=0.2, color='red')
    
    axes[1].set_xlabel('Distance from Code 0 (binned)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Spectral Correlation', fontsize=13, fontweight='bold')
    axes[1].set_title('Binned Spectral Correlation', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
    
    plt.suptitle(f'Binned Analysis - Base Code 0 ({num_bins} bins)', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_dir / 'code0_correlation_vs_distance_binned.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Binned plot salvato: {save_dir / 'code0_correlation_vs_distance_binned.png'}")
    
    # ========================================================================
    # PLOT 3: Distribuzione delle correlazioni
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram temporal correlation
    axes[0, 0].hist(temp_corr, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(temp_corr), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(temp_corr):.3f}')
    axes[0, 0].axvline(np.median(temp_corr), color='orange', linestyle='--',
                      linewidth=2, label=f'Median: {np.median(temp_corr):.3f}')
    axes[0, 0].set_xlabel('Temporal Correlation', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of Temporal Correlations', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Histogram spectral correlation
    axes[0, 1].hist(spec_corr, bins=30, color='red', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(spec_corr), color='blue', linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(spec_corr):.3f}')
    axes[0, 1].axvline(np.median(spec_corr), color='orange', linestyle='--',
                      linewidth=2, label=f'Median: {np.median(spec_corr):.3f}')
    axes[0, 1].set_xlabel('Spectral Correlation', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Distribution of Spectral Correlations', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Histogram distances
    axes[1, 0].hist(distances, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(distances), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(distances):.3f}')
    axes[1, 0].set_xlabel('Distance from Code 0', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Distribution of Distances', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Scatter: Temporal vs Spectral correlation
    axes[1, 1].scatter(temp_corr, spec_corr, alpha=0.6, s=50,
                      c=distances, cmap='viridis', edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Temporal Correlation', fontsize=12)
    axes[1, 1].set_ylabel('Spectral Correlation', fontsize=12)
    axes[1, 1].set_title('Temporal vs Spectral Correlation', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Correlation between temporal and spectral
    corr_temp_spec, _ = pearsonr(temp_corr, spec_corr)
    axes[1, 1].text(0.05, 0.95, f'Corr(temp, spec) = {corr_temp_spec:.3f}',
                   transform=axes[1, 1].transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Distance')
    
    plt.suptitle('Distribution Analysis - Base Code 0', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'code0_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Distribution plot salvato: {save_dir / 'code0_distributions.png'}")
    
    # ========================================================================
    # PLOT 4: Top/Bottom codici
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top 5 più vicini (bassa distanza)
    top_5_closest_idx = np.argsort(distances)[:5]
    
    axes[0, 0].bar(range(5), distances[top_5_closest_idx], color='green', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xticks(range(5))
    axes[0, 0].set_xticklabels([f'Code {code_indices[i]}' for i in top_5_closest_idx], rotation=45)
    axes[0, 0].set_ylabel('Distance', fontsize=12)
    axes[0, 0].set_title('Top 5 Closest Codes', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Correlazioni dei più vicini
    x = np.arange(5)
    width = 0.35
    axes[0, 1].bar(x - width/2, temp_corr[top_5_closest_idx], width,
                  label='Temporal', color='blue', alpha=0.7, edgecolor='black')
    axes[0, 1].bar(x + width/2, spec_corr[top_5_closest_idx], width,
                  label='Spectral', color='red', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'Code {code_indices[i]}' for i in top_5_closest_idx], rotation=45)
    axes[0, 1].set_ylabel('Correlation', fontsize=12)
    axes[0, 1].set_title('Correlations of Closest Codes', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Top 5 più lontani (alta distanza)
    top_5_farthest_idx = np.argsort(distances)[-5:][::-1]
    
    axes[1, 0].bar(range(5), distances[top_5_farthest_idx], color='red', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xticks(range(5))
    axes[1, 0].set_xticklabels([f'Code {code_indices[i]}' for i in top_5_farthest_idx], rotation=45)
    axes[1, 0].set_ylabel('Distance', fontsize=12)
    axes[1, 0].set_title('Top 5 Farthest Codes', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Correlazioni dei più lontani
    axes[1, 1].bar(x - width/2, temp_corr[top_5_farthest_idx], width,
                  label='Temporal', color='blue', alpha=0.7, edgecolor='black')
    axes[1, 1].bar(x + width/2, spec_corr[top_5_farthest_idx], width,
                  label='Spectral', color='red', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'Code {code_indices[i]}' for i in top_5_farthest_idx], rotation=45)
    axes[1, 1].set_ylabel('Correlation', fontsize=12)
    axes[1, 1].set_title('Correlations of Farthest Codes', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Closest vs Farthest Codes Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'code0_top_bottom_codes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Top/Bottom codes plot salvato: {save_dir / 'code0_top_bottom_codes.png'}")
    
    return {
        'corr_temporal_vs_distance': corr_temp_dist,
        'corr_spectral_vs_distance': corr_spec_dist,
        'p_value_temporal': p_temp_val,
        'p_value_spectral': p_spec_val
    }


def main():
    print("="*70)
    print("📊 ANALISI CORRELAZIONE vs DISTANZA - CODICE BASE 0")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Temporal length: {TEMPORAL_LENGTH}")
    print(f"   Base code: {BASE_CODE}")
    print(f"   W&B Project: {WANDB_PROJECT}")
    
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
            **MODEL_CONFIG
        },
        tags=["correlation", "distance-analysis", "code0"]
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
        
        # 3. Analizza codice 0
        print(f"\n{'='*70}")
        print(f"📊 ANALISI CODICE BASE 0")
        print(f"{'='*70}")
        
        results = analyze_code0_correlations_vs_distance(
            model, embeddings, device, TEMPORAL_LENGTH
        )
        
        # Log metriche base
        wandb.log({
            'mean_temporal_corr': np.mean(results['temporal_correlations']),
            'std_temporal_corr': np.std(results['temporal_correlations']),
            'mean_spectral_corr': np.mean(results['spectral_correlations']),
            'std_spectral_corr': np.std(results['spectral_correlations']),
            'mean_distance': np.mean(results['distances']),
            'std_distance': np.std(results['distances']),
        })
        
        # 4. Visualizza risultati
        print(f"\n{'='*70}")
        print("📈 CREAZIONE VISUALIZZAZIONI")
        print(f"{'='*70}")
        
        viz_stats = visualize_code0_correlation_vs_distance(results, SAVE_DIR)
        
        # Log immagini su W&B
        wandb.log({
            "main_plot": wandb.Image(str(SAVE_DIR / 'code0_correlation_vs_distance_main.png')),
            "binned_plot": wandb.Image(str(SAVE_DIR / 'code0_correlation_vs_distance_binned.png')),
            "distributions": wandb.Image(str(SAVE_DIR / 'code0_distributions.png')),
            "top_bottom_codes": wandb.Image(str(SAVE_DIR / 'code0_top_bottom_codes.png')),
        })
        
        # 5. Statistiche finali
        print(f"\n{'='*70}")
        print("📊 STATISTICHE FINALI")
        print(f"{'='*70}")
        
        print(f"\n   🔵 Temporal Correlation:")
        print(f"      Mean: {np.mean(results['temporal_correlations']):.4f}")
        print(f"      Std: {np.std(results['temporal_correlations']):.4f}")
        print(f"      Min: {np.min(results['temporal_correlations']):.4f}")
        print(f"      Max: {np.max(results['temporal_correlations']):.4f}")
        print(f"      Corr with Distance: {viz_stats['corr_temporal_vs_distance']:.4f}")
        print(f"      P-value: {viz_stats['p_value_temporal']:.4e}")
        
        print(f"\n   🔴 Spectral Correlation:")
        print(f"      Mean: {np.mean(results['spectral_correlations']):.4f}")
        print(f"      Std: {np.std(results['spectral_correlations']):.4f}")
        print(f"      Min: {np.min(results['spectral_correlations']):.4f}")
        print(f"      Max: {np.max(results['spectral_correlations']):.4f}")
        print(f"      Corr with Distance: {viz_stats['corr_spectral_vs_distance']:.4f}")
        print(f"      P-value: {viz_stats['p_value_spectral']:.4e}")
        
        print(f"\n   🟢 Distance:")
        print(f"      Mean: {np.mean(results['distances']):.4f}")
        print(f"      Std: {np.std(results['distances']):.4f}")
        print(f"      Min: {np.min(results['distances']):.4f}")
        print(f"      Max: {np.max(results['distances']):.4f}")
        
        # Log statistiche aggregate
        wandb.log({
            'corr_temporal_vs_distance': viz_stats['corr_temporal_vs_distance'],
            'corr_spectral_vs_distance': viz_stats['corr_spectral_vs_distance'],
            'p_value_temporal': viz_stats['p_value_temporal'],
            'p_value_spectral': viz_stats['p_value_spectral'],
        })
        
        # 6. Interpretazione
        print(f"\n{'='*70}")
        print("💡 INTERPRETAZIONE")
        print(f"{'='*70}")
        
        if viz_stats['corr_temporal_vs_distance'] < -0.3 and viz_stats['p_value_temporal'] < 0.05:
            print(f"\n   ✅ TEMPORAL: Strong negative correlation (r={viz_stats['corr_temporal_vs_distance']:.3f}, p<0.05)")
            print(f"      → Codici più distanti hanno tracce temporali meno correlate")
        elif viz_stats['corr_temporal_vs_distance'] < -0.1:
            print(f"\n   ⚠️  TEMPORAL: Weak negative correlation (r={viz_stats['corr_temporal_vs_distance']:.3f})")
            print(f"      → Trend debole di decorrelazione temporale con distanza")
        else:
            print(f"\n   ❌ TEMPORAL: No clear relationship (r={viz_stats['corr_temporal_vs_distance']:.3f})")
            print(f"      → Distanza embedding non predice correlazione temporale")
        
        if viz_stats['corr_spectral_vs_distance'] < -0.3 and viz_stats['p_value_spectral'] < 0.05:
            print(f"\n   ✅ SPECTRAL: Strong negative correlation (r={viz_stats['corr_spectral_vs_distance']:.3f}, p<0.05)")
            print(f"      → Codici più distanti hanno spettri meno correlati")
        elif viz_stats['corr_spectral_vs_distance'] < -0.1:
            print(f"\n   ⚠️  SPECTRAL: Weak negative correlation (r={viz_stats['corr_spectral_vs_distance']:.3f})")
            print(f"      → Trend debole di decorrelazione spettrale con distanza")
        else:
            print(f"\n   ❌ SPECTRAL: No clear relationship (r={viz_stats['corr_spectral_vs_distance']:.3f})")
            print(f"      → Distanza embedding non predice correlazione spettrale")
        
        # 7. Summary finale
        print(f"\n{'='*70}")
        print("✅ ANALISI COMPLETATA!")
        print(f"{'='*70}")
        print(f"\n📁 Risultati salvati in: {SAVE_DIR}/")
        print(f"🌐 W&B: {wandb.run.url}")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        wandb.finish()
        print("\n✅ W&B run completato!")


if __name__ == "__main__":
    main()