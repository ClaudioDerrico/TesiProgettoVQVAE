#!/usr/bin/env python3
"""
Analisi Temporale della Similarità del Codebook DUAL

Risponde alla domanda: "Perché due codici sono vicini/lontani nel clustering?"
Per entrambi i codebook (active e inactive).

Strategia:
1. Clustering gerarchico degli embeddings (separato per active/inactive)
2. Identifica coppie di codici:
   - VICINI: stessa branch (alta similarità)
   - LONTANI: branch opposte (bassa similarità)
3. Decodifica le tracce temporali
4. Visualizza e confronta per capire COSA li rende simili/diversi

Uso:
    python codebook_similarity_temporal_analysis_dual.py
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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

from models.dual_vqvae import DualCalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/dual_codebook/best_dual_model_32.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-codebook-similarity-temporal-dual"
WANDB_RUN_NAME = f"similarity-temporal-dual-{datetime.now().strftime('%m%d-%H%M')}"

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
    'threshold': 0.0,
    'threshold_type': 'adaptive',
}

# Parametri
TEMPORAL_LENGTH = 15
NUM_NEIGHBOR_PAIRS = 3  # Quante coppie di vicini analizzare
NUM_DISTANT_PAIRS = 3   # Quante coppie di lontani analizzare

# ============================================================================
# FUNZIONI
# ============================================================================

def load_model(checkpoint_path, config, device):
    """Carica il modello dual allenato"""
    print(f"🔄 Caricando modello DUAL da: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Auto-infer num_embeddings se necessario
    if 'model_config' in checkpoint:
        model_config_checkpoint = checkpoint['model_config']
        if 'num_embeddings' in model_config_checkpoint:
            config['num_embeddings'] = model_config_checkpoint['num_embeddings']
    
    model = DualCalciumVQVAE(
        num_neurons=config['num_neurons'],
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        dropout_rate=config['dropout_rate'],
        use_quantizer=config['use_quantizer'],
        threshold=config.get('threshold', 0.0),
        threshold_type=config.get('threshold_type', 'adaptive')
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Modello DUAL caricato (epoca {checkpoint.get('epoch', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Modello DUAL caricato")
    
    model = model.to(device)
    model.eval()
    
    return model


def extract_codebook_embeddings_dual(model):
    """Estrae gli embeddings di entrambi i codebook"""
    vq = model.vector_quantization
    embeddings_active = vq.embedding_active.weight.data.cpu().numpy()
    embeddings_inactive = vq.embedding_inactive.weight.data.cpu().numpy()
    
    print(f"✅ Codebook embeddings estratti:")
    print(f"   Active: {embeddings_active.shape}")
    print(f"   Inactive: {embeddings_inactive.shape}")
    
    return embeddings_active, embeddings_inactive


def decode_code_dual(model, code_idx, codebook_type, device, temporal_length):
    """
    Decodifica un singolo codice dal codebook specificato
    
    Args:
        code_idx: indice del codice nel codebook specificato
        codebook_type: 'active' o 'inactive'
    
    Returns:
        trace: (output_time,) - traccia temporale del neurone
    """
    vq = model.vector_quantization
    
    if codebook_type == 'active':
        code_embedding = vq.embedding_active.weight[code_idx]
    elif codebook_type == 'inactive':
        code_embedding = vq.embedding_inactive.weight[code_idx]
    else:
        raise ValueError(f"codebook_type deve essere 'active' o 'inactive', ricevuto: {codebook_type}")
    
    # Ripeti per temporal_length
    sequence = code_embedding.unsqueeze(1).repeat(1, temporal_length)
    sequence = sequence.unsqueeze(0).to(device)
    
    # Decodifica
    with torch.no_grad():
        reconstruction = model.decode(sequence)
    
    trace = reconstruction.cpu().numpy()[0, 0, :]  # (output_time,)
    
    return trace


def find_neighbor_and_distant_pairs(embeddings, linkage_matrix, num_neighbor_pairs, num_distant_pairs):
    """
    Identifica coppie di codici vicini e lontani usando il dendrogramma
    
    Args:
        embeddings: (num_codes, embedding_dim)
        linkage_matrix: output di scipy.cluster.hierarchy.linkage
        num_neighbor_pairs: numero di coppie vicine da trovare
        num_distant_pairs: numero di coppie distanti da trovare
    
    Returns:
        neighbor_pairs: lista di (code_a, code_b, distance, similarity)
        distant_pairs: lista di (code_a, code_b, distance, similarity)
    """
    
    num_codes = embeddings.shape[0]
    
    # Calcola matrice di similarità
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    
    # ========================================================================
    # TROVA COPPIE VICINE
    # ========================================================================
    neighbor_pairs = []
    
    for i in range(num_codes):
        for j in range(i + 1, num_codes):
            dist = distance_matrix[i, j]
            sim = similarity_matrix[i, j]
            neighbor_pairs.append((i, j, dist, sim))
    
    # Ordina per distanza crescente (i più vicini per primi)
    neighbor_pairs.sort(key=lambda x: x[2])
    top_neighbor_pairs = neighbor_pairs[:num_neighbor_pairs]
    
    # ========================================================================
    # TROVA COPPIE DISTANTI
    # ========================================================================
    distant_pairs_sorted = sorted(neighbor_pairs, key=lambda x: x[2], reverse=True)
    top_distant_pairs = distant_pairs_sorted[:num_distant_pairs]
    
    print(f"\n📊 Coppie identificate:")
    print(f"\n   🟢 VICINE (top {num_neighbor_pairs}):")
    for i, (code_a, code_b, dist, sim) in enumerate(top_neighbor_pairs, 1):
        print(f"      {i}. Codes {code_a}-{code_b}: distance={dist:.4f}, similarity={sim:.4f}")
    
    print(f"\n   🔴 DISTANTI (top {num_distant_pairs}):")
    for i, (code_a, code_b, dist, sim) in enumerate(top_distant_pairs, 1):
        print(f"      {i}. Codes {code_a}-{code_b}: distance={dist:.4f}, similarity={sim:.4f}")
    
    return top_neighbor_pairs, top_distant_pairs


# Riutilizza le funzioni di analisi dal file originale
def analyze_trace_characteristics(trace):
    """Analizza caratteristiche di una traccia temporale"""
    characteristics = {
        'amplitude': np.max(trace) - np.min(trace),
        'mean': np.mean(trace),
        'std': np.std(trace),
        'variance': np.var(trace),
        'max': np.max(trace),
        'min': np.min(trace),
        'energy': np.sum(trace ** 2),
        'range': np.ptp(trace),
    }
    
    threshold = characteristics['mean'] + 2 * characteristics['std']
    peak_mask = trace > threshold
    if len(peak_mask) > 1:
        num_peaks = np.sum(np.diff(peak_mask.astype(int)) == 1)
    else:
        num_peaks = 0
    characteristics['num_peaks'] = num_peaks
    
    time_points = np.arange(len(trace))
    if len(time_points) > 1:
        slope, _ = np.polyfit(time_points, trace, 1)
    else:
        slope = 0
    characteristics['trend_slope'] = slope
    
    return characteristics


def compute_fft_spectrum(trace, sample_rate=1.0):
    """Calcola lo spettro di frequenza usando FFT"""
    n = len(trace)
    trace_centered = trace - np.mean(trace)
    fft_vals = np.fft.fft(trace_centered)
    freqs = np.fft.fftfreq(n, d=1.0/sample_rate)
    positive_freq_idx = freqs >= 0
    freqs_positive = freqs[positive_freq_idx]
    fft_positive = fft_vals[positive_freq_idx]
    magnitude = np.abs(fft_positive)
    power = magnitude ** 2
    phase = np.angle(fft_positive)
    return freqs_positive, power, magnitude, phase


def analyze_spectral_similarity(trace_a, trace_b):
    """Analizza similarità spettrale tra due tracce"""
    freqs_a, power_a, mag_a, phase_a = compute_fft_spectrum(trace_a)
    freqs_b, power_b, mag_b, phase_b = compute_fft_spectrum(trace_b)
    
    power_a_norm = power_a / (np.sum(power_a) + 1e-8)
    power_b_norm = power_b / (np.sum(power_b) + 1e-8)
    
    if len(power_a_norm) > 1 and np.std(power_a_norm) > 1e-8 and np.std(power_b_norm) > 1e-8:
        spectral_corr, _ = pearsonr(power_a_norm, power_b_norm)
    else:
        spectral_corr = 0.0
    
    spectral_l2_dist = np.sqrt(np.sum((power_a_norm - power_b_norm) ** 2))
    
    if len(power_a) > 1:
        dominant_freq_a_idx = np.argmax(power_a[1:]) + 1
        dominant_freq_a = freqs_a[dominant_freq_a_idx]
        dominant_power_a = power_a[dominant_freq_a_idx]
    else:
        dominant_freq_a = 0
        dominant_power_a = 0
    
    if len(power_b) > 1:
        dominant_freq_b_idx = np.argmax(power_b[1:]) + 1
        dominant_freq_b = freqs_b[dominant_freq_b_idx]
        dominant_power_b = power_b[dominant_freq_b_idx]
    else:
        dominant_freq_b = 0
        dominant_power_b = 0
    
    freq_diff = abs(dominant_freq_a - dominant_freq_b)
    
    threshold_a = 0.1 * np.max(power_a)
    threshold_b = 0.1 * np.max(power_b)
    bandwidth_a = np.sum(power_a > threshold_a)
    bandwidth_b = np.sum(power_b > threshold_b)
    
    return {
        'spectral_correlation': spectral_corr,
        'spectral_l2_distance': spectral_l2_dist,
        'dominant_freq_a': dominant_freq_a,
        'dominant_freq_b': dominant_freq_b,
        'dominant_freq_diff': freq_diff,
        'dominant_power_a': dominant_power_a,
        'dominant_power_b': dominant_power_b,
        'bandwidth_a': bandwidth_a,
        'bandwidth_b': bandwidth_b,
        'freqs': freqs_a,
        'power_a': power_a,
        'power_b': power_b,
        'power_a_norm': power_a_norm,
        'power_b_norm': power_b_norm,
    }


def visualize_code_pair_comparison_dual(code_a, code_b, trace_a, trace_b, 
                                       similarity, distance, pair_type, codebook_type):
    """
    Visualizza confronto dettagliato tra due codici (TEMPO + FREQUENZA) per DUAL codebook
    
    Args:
        codebook_type: 'active' o 'inactive'
    """
    
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)
    
    time_points = np.arange(len(trace_a))
    
    chars_a = analyze_trace_characteristics(trace_a)
    chars_b = analyze_trace_characteristics(trace_b)
    spectral_analysis = analyze_spectral_similarity(trace_a, trace_b)
    
    # ROW 1: Tracce temporali individuali
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax1.plot(time_points, trace_a, 'b-', linewidth=2, label=f'Code {code_a}')
    ax1.axhline(chars_a['mean'], color='cyan', linestyle='--', alpha=0.5, label=f"Mean: {chars_a['mean']:.3f}")
    ax1.fill_between(time_points, chars_a['mean'] - chars_a['std'], chars_a['mean'] + chars_a['std'],
                     alpha=0.2, color='blue', label=f"±1 STD")
    ax1.set_ylabel('Activity')
    ax1.set_title(f'Code {code_a} Temporal Trace', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    stats_text_a = f"Amplitude: {chars_a['amplitude']:.3f}\n"
    stats_text_a += f"Mean: {chars_a['mean']:.3f}\n"
    stats_text_a += f"Std: {chars_a['std']:.3f}\n"
    stats_text_a += f"Max: {chars_a['max']:.3f}\n"
    stats_text_a += f"Peaks: {chars_a['num_peaks']}\n"
    stats_text_a += f"Energy: {chars_a['energy']:.3f}"
    
    ax1.text(0.02, 0.98, stats_text_a, transform=ax1.transAxes,
             verticalalignment='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax2.plot(time_points, trace_b, 'r-', linewidth=2, label=f'Code {code_b}')
    ax2.axhline(chars_b['mean'], color='orange', linestyle='--', alpha=0.5, label=f"Mean: {chars_b['mean']:.3f}")
    ax2.fill_between(time_points, chars_b['mean'] - chars_b['std'], chars_b['mean'] + chars_b['std'],
                     alpha=0.2, color='red', label=f"±1 STD")
    ax2.set_ylabel('Activity')
    ax2.set_title(f'Code {code_b} Temporal Trace', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    stats_text_b = f"Amplitude: {chars_b['amplitude']:.3f}\n"
    stats_text_b += f"Mean: {chars_b['mean']:.3f}\n"
    stats_text_b += f"Std: {chars_b['std']:.3f}\n"
    stats_text_b += f"Max: {chars_b['max']:.3f}\n"
    stats_text_b += f"Peaks: {chars_b['num_peaks']}\n"
    stats_text_b += f"Energy: {chars_b['energy']:.3f}"
    
    ax2.text(0.02, 0.98, stats_text_b, transform=ax2.transAxes,
             verticalalignment='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # ROW 2: Spettri di frequenza
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    freqs = spectral_analysis['freqs']
    power_a = spectral_analysis['power_a']
    power_b = spectral_analysis['power_b']
    
    ax3.plot(freqs, power_a, 'b-', linewidth=2, label=f'Code {code_a}')
    ax3.axvline(spectral_analysis['dominant_freq_a'], color='cyan', linestyle='--', linewidth=2,
                label=f"Dom. Freq: {spectral_analysis['dominant_freq_a']:.3f} Hz")
    ax3.fill_between(freqs, 0, power_a, alpha=0.3, color='blue')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power')
    ax3.set_title(f'Code {code_a} Power Spectrum', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, np.max(freqs) * 0.5])
    
    ax4.plot(freqs, power_b, 'r-', linewidth=2, label=f'Code {code_b}')
    ax4.axvline(spectral_analysis['dominant_freq_b'], color='orange', linestyle='--', linewidth=2,
                label=f"Dom. Freq: {spectral_analysis['dominant_freq_b']:.3f} Hz")
    ax4.fill_between(freqs, 0, power_b, alpha=0.3, color='red')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power')
    ax4.set_title(f'Code {code_b} Power Spectrum', fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, np.max(freqs) * 0.5])
    
    # ROW 3: Overlay spettri + differenza spettrale
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    
    power_a_norm = spectral_analysis['power_a_norm']
    power_b_norm = spectral_analysis['power_b_norm']
    
    ax5.plot(freqs, power_a_norm, 'b-', linewidth=2, alpha=0.7, label=f'Code {code_a}')
    ax5.plot(freqs, power_b_norm, 'r-', linewidth=2, alpha=0.7, label=f'Code {code_b}')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Normalized Power')
    ax5.set_title(f'Spectral Overlay (Normalized)\n'
                  f'Spectral Correlation: {spectral_analysis["spectral_correlation"]:.3f}',
                  fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, np.max(freqs) * 0.5])
    
    spectral_diff = np.abs(power_a_norm - power_b_norm)
    ax6.plot(freqs, spectral_diff, 'k-', linewidth=2, label='|Spectral Difference|')
    ax6.fill_between(freqs, 0, spectral_diff, alpha=0.3, color='purple')
    ax6.axhline(np.mean(spectral_diff), color='red', linestyle='--',
                label=f'Mean: {np.mean(spectral_diff):.4f}')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Absolute Difference')
    ax6.set_title(f'Spectral Difference\n'
                  f'L2 Distance: {spectral_analysis["spectral_l2_distance"]:.4f}',
                  fontweight='bold')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, np.max(freqs) * 0.5])
    
    # ROW 4: Overlay temporale + Feature comparison
    ax7 = fig.add_subplot(gs[3, 0])
    ax8 = fig.add_subplot(gs[3, 1])
    
    ax7.plot(time_points, trace_a, 'b-', linewidth=2, alpha=0.7, label=f'Code {code_a}')
    ax7.plot(time_points, trace_b, 'r-', linewidth=2, alpha=0.7, label=f'Code {code_b}')
    
    diff = trace_b - trace_a
    ax7.fill_between(time_points, trace_a, trace_b, where=(diff > 0), color='orange', alpha=0.2, label='B > A')
    ax7.fill_between(time_points, trace_a, trace_b, where=(diff < 0), color='purple', alpha=0.2, label='A > B')
    
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Activity')
    
    if np.std(trace_a) > 1e-8 and np.std(trace_b) > 1e-8:
        temporal_corr, _ = pearsonr(trace_a, trace_b)
    else:
        temporal_corr = 0.0
    
    ax7.set_title(f'Temporal Overlay (Corr: {temporal_corr:.3f})', fontweight='bold')
    ax7.legend(loc='best')
    ax7.grid(True, alpha=0.3)
    
    # Feature comparison
    feature_names = ['amplitude', 'mean', 'energy', 'num_peaks', 
                     'dom_freq', 'bandwidth', 'spectral_corr']
    
    features_a = [
        chars_a['amplitude'],
        chars_a['mean'],
        chars_a['energy'],
        chars_a['num_peaks'],
        spectral_analysis['dominant_freq_a'],
        spectral_analysis['bandwidth_a'],
        spectral_analysis['spectral_correlation']
    ]
    
    features_b = [
        chars_b['amplitude'],
        chars_b['mean'],
        chars_b['energy'],
        chars_b['num_peaks'],
        spectral_analysis['dominant_freq_b'],
        spectral_analysis['bandwidth_b'],
        spectral_analysis['spectral_correlation']
    ]
    
    features_a_norm = []
    features_b_norm = []
    for i, (fa, fb) in enumerate(zip(features_a, features_b)):
        if feature_names[i] == 'spectral_corr':
            features_a_norm.append(fa)
            features_b_norm.append(fb)
        else:
            min_val = min(fa, fb)
            max_val = max(fa, fb)
            if max_val - min_val > 1e-8:
                features_a_norm.append((fa - min_val) / (max_val - min_val))
                features_b_norm.append((fb - min_val) / (max_val - min_val))
            else:
                features_a_norm.append(0.5)
                features_b_norm.append(0.5)
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    ax8.bar(x - width/2, features_a_norm, width, label=f'Code {code_a}', color='blue', alpha=0.7)
    ax8.bar(x + width/2, features_b_norm, width, label=f'Code {code_b}', color='red', alpha=0.7)
    
    ax8.set_xlabel('Feature')
    ax8.set_ylabel('Normalized Value')
    ax8.set_title('Feature Comparison (Temporal + Spectral)', fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(feature_names, rotation=45, ha='right')
    ax8.legend(loc='best')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Title
    feature_diffs = [abs(features_a[i] - features_b[i]) for i in range(len(features_a))]
    sorted_diffs = sorted(enumerate(feature_diffs), key=lambda x: x[1])
    most_similar_features = [feature_names[i] for i, _ in sorted_diffs[:3]]
    most_different_features = [feature_names[i] for i, _ in sorted_diffs[-3:]]
    
    title_color = 'green' if pair_type == "NEIGHBOR" else 'red'
    codebook_color = 'red' if codebook_type == 'active' else 'blue'
    
    fig.suptitle(
        f'{pair_type} PAIR ({codebook_type.upper()} CODEBOOK): Codes {code_a} ↔ {code_b}\n'
        f'EMBEDDING: Similarity={similarity:.4f}, Distance={distance:.4f}\n'
        f'TEMPORAL: Correlation={temporal_corr:.3f} | '
        f'SPECTRAL: Correlation={spectral_analysis["spectral_correlation"]:.3f}, '
        f'Freq Diff={spectral_analysis["dominant_freq_diff"]:.3f} Hz\n'
        f'Most Similar: {", ".join(most_similar_features)} | '
        f'Most Different: {", ".join(most_different_features)}',
        fontsize=13, fontweight='bold', color=title_color
    )
    
    return fig


def analyze_why_similar_or_different(chars_a, chars_b):
    """Analizza quali caratteristiche rendono due codici simili o diversi"""
    feature_names = ['amplitude', 'mean', 'std', 'variance', 'max', 'min', 
                     'energy', 'num_peaks', 'trend_slope']
    
    abs_diffs = {}
    norm_diffs = {}
    
    for feat in feature_names:
        val_a = chars_a[feat]
        val_b = chars_b[feat]
        abs_diff = abs(val_a - val_b)
        abs_diffs[feat] = abs_diff
        if abs(val_a) > 1e-8:
            norm_diff = abs_diff / abs(val_a)
        else:
            norm_diff = abs_diff
        norm_diffs[feat] = norm_diff
    
    sorted_features = sorted(norm_diffs.items(), key=lambda x: x[1])
    
    return {
        'most_similar_features': [feat for feat, _ in sorted_features[:3]],
        'most_different_features': [feat for feat, _ in sorted_features[-3:]],
        'abs_diffs': abs_diffs,
        'norm_diffs': norm_diffs,
        'sorted_features': sorted_features
    }


def main():
    print("="*70)
    print("🔬 ANALISI TEMPORALE SIMILARITÀ CODEBOOK DUAL")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Temporal length: {TEMPORAL_LENGTH}")
    print(f"   Num neighbor pairs: {NUM_NEIGHBOR_PAIRS}")
    print(f"   Num distant pairs: {NUM_DISTANT_PAIRS}")
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
            "num_neighbor_pairs": NUM_NEIGHBOR_PAIRS,
            "num_distant_pairs": NUM_DISTANT_PAIRS,
            **MODEL_CONFIG
        },
        tags=["similarity", "temporal-analysis", "clustering", "codebook", "dual"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # 1. Carica modello
        model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # 2. Estrai embeddings
        embeddings_active, embeddings_inactive = extract_codebook_embeddings_dual(model)
        
        # 3. Analizza per entrambi i codebook
        for codebook_type, embeddings in [('active', embeddings_active), ('inactive', embeddings_inactive)]:
            print("\n" + "="*70)
            print(f"🔬 ANALISI CODEBOOK: {codebook_type.upper()}")
            print("="*70)
            
            # Clustering gerarchico
            print(f"\n🔬 Eseguendo clustering gerarchico per {codebook_type}...")
            similarity_matrix = cosine_similarity(embeddings)
            distance_matrix = 1 - similarity_matrix
            condensed_dist = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_dist, method='average')
            print("✅ Clustering completato")
            
            # Trova coppie vicine e distanti
            print(f"\n🔍 Identificando coppie di codici per {codebook_type}...")
            neighbor_pairs, distant_pairs = find_neighbor_and_distant_pairs(
                embeddings, linkage_matrix, NUM_NEIGHBOR_PAIRS, NUM_DISTANT_PAIRS
            )
            
            # Analizza coppie VICINE
            print("\n" + "="*70)
            print(f"🟢 ANALISI COPPIE VICINE ({codebook_type.upper()})")
            print("="*70)
            
            for pair_idx, (code_a, code_b, dist, sim) in enumerate(neighbor_pairs, 1):
                print(f"\n📊 Coppia {pair_idx}: Codes {code_a} ↔ {code_b}")
                print(f"   Distance: {dist:.4f}, Similarity: {sim:.4f}")
                
                trace_a = decode_code_dual(model, code_a, codebook_type, device, TEMPORAL_LENGTH)
                trace_b = decode_code_dual(model, code_b, codebook_type, device, TEMPORAL_LENGTH)
                
                chars_a = analyze_trace_characteristics(trace_a)
                chars_b = analyze_trace_characteristics(trace_b)
                spectral_analysis = analyze_spectral_similarity(trace_a, trace_b)
                analysis = analyze_why_similar_or_different(chars_a, chars_b)
                
                print(f"\n   🔍 Perché sono SIMILI?")
                print(f"      TEMPORAL - Most similar features: {', '.join(analysis['most_similar_features'])}")
                print(f"\n      SPECTRAL - Frequency domain analysis:")
                print(f"         Spectral Correlation: {spectral_analysis['spectral_correlation']:.4f}")
                print(f"         Dominant Freq A: {spectral_analysis['dominant_freq_a']:.4f} Hz")
                print(f"         Dominant Freq B: {spectral_analysis['dominant_freq_b']:.4f} Hz")
                print(f"         Frequency Difference: {spectral_analysis['dominant_freq_diff']:.4f} Hz")
                
                fig = visualize_code_pair_comparison_dual(
                    code_a, code_b, trace_a, trace_b, sim, dist, "NEIGHBOR", codebook_type
                )
                
                wandb.log({f"{codebook_type}/neighbor_pairs/pair_{pair_idx}_codes_{code_a}_{code_b}": wandb.Image(fig)})
                plt.close(fig)
            
            # Analizza coppie DISTANTI
            print("\n" + "="*70)
            print(f"🔴 ANALISI COPPIE DISTANTI ({codebook_type.upper()})")
            print("="*70)
            
            for pair_idx, (code_a, code_b, dist, sim) in enumerate(distant_pairs, 1):
                print(f"\n📊 Coppia {pair_idx}: Codes {code_a} ↔ {code_b}")
                print(f"   Distance: {dist:.4f}, Similarity: {sim:.4f}")
                
                trace_a = decode_code_dual(model, code_a, codebook_type, device, TEMPORAL_LENGTH)
                trace_b = decode_code_dual(model, code_b, codebook_type, device, TEMPORAL_LENGTH)
                
                chars_a = analyze_trace_characteristics(trace_a)
                chars_b = analyze_trace_characteristics(trace_b)
                spectral_analysis = analyze_spectral_similarity(trace_a, trace_b)
                analysis = analyze_why_similar_or_different(chars_a, chars_b)
                
                print(f"\n   🔍 Perché sono DIVERSI?")
                print(f"      TEMPORAL - Most different features: {', '.join(analysis['most_different_features'])}")
                print(f"\n      SPECTRAL - Frequency domain analysis:")
                print(f"         Spectral Correlation: {spectral_analysis['spectral_correlation']:.4f}")
                print(f"         Dominant Freq A: {spectral_analysis['dominant_freq_a']:.4f} Hz")
                print(f"         Dominant Freq B: {spectral_analysis['dominant_freq_b']:.4f} Hz")
                print(f"         Frequency Difference: {spectral_analysis['dominant_freq_diff']:.4f} Hz")
                
                fig = visualize_code_pair_comparison_dual(
                    code_a, code_b, trace_a, trace_b, sim, dist, "DISTANT", codebook_type
                )
                
                wandb.log({f"{codebook_type}/distant_pairs/pair_{pair_idx}_codes_{code_a}_{code_b}": wandb.Image(fig)})
                plt.close(fig)
        
        # Summary
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        
        print(f"\n✅ Analizzate {NUM_NEIGHBOR_PAIRS} coppie vicine per codebook")
        print(f"✅ Analizzate {NUM_DISTANT_PAIRS} coppie distanti per codebook")
        print(f"\n🌐 Tutti i risultati su W&B: {wandb.run.url}")
        
        print("\n" + "="*70)
        print("✅ ANALISI COMPLETATA!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        wandb.finish()
        print("✅ W&B run completato!")


if __name__ == "__main__":
    main()


