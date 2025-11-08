#!/usr/bin/env python3
"""
Visualizzazione Scatter: Distanze vs Correlazioni

Crea 4 scatter plot che mostrano la relazione tra:
- Distanze (Euclidea, Coseno) sull'asse X
- Correlazioni (Temporale, Spettrale) sull'asse Y

Uso:
    python plot_distance_correlation_scatter.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
import wandb

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

SUMMARY_DIR = "correlazione_distanze_temporali_spettrali/sommario coppie"

# Configurazione W&B
WANDB_PROJECT = "vqvae-analysis"
WANDB_RUN_NAME = "distance-correlation-scatter"
USE_WANDB = True  # Set to False per disabilitare W&B

# ============================================================================
# FUNZIONI - CARICAMENTO DATI
# ============================================================================

def load_all_summaries(summary_dir):
    """
    Carica tutti i file summary_codice_*.json dalla cartella
    
    Returns:
        dict: {code_idx: summary_data}
    """
    summary_dir_path = Path(summary_dir)
    
    if not summary_dir_path.exists():
        raise FileNotFoundError(f"Directory {summary_dir} non trovata!")
    
    # Prova diversi pattern
    summary_files = list(summary_dir_path.glob("summary_codice_*.json"))
    
    if not summary_files:
        # Prova anche senza underscore
        summary_files = list(summary_dir_path.glob("summary_codice*.json"))
    
    if not summary_files:
        # Lista tutti i json per debug
        all_json = list(summary_dir_path.glob("*.json"))
        print(f"⚠️  File JSON trovati nella directory:")
        for f in all_json:
            print(f"    - {f.name}")
        raise FileNotFoundError(f"Nessun file summary trovato in {summary_dir}/")
    
    summaries = {}
    
    for file_path in summary_files:
        # Estrai il numero del codice dal nome file
        filename = file_path.stem
        
        # Prova a estrarre il numero
        try:
            # Rimuovi "summary_codice_" o "summary_codice"
            if '_' in filename:
                parts = filename.split('_')
                code_idx = int(parts[-1])
            else:
                import re
                match = re.search(r'(\d+)$', filename)
                if match:
                    code_idx = int(match.group(1))
                else:
                    raise ValueError(f"Cannot extract code index from {filename}")
        except Exception as e:
            print(f"⚠️  Skipping file {file_path.name}: {e}")
            continue
        
        with open(file_path, 'r') as f:
            summaries[code_idx] = json.load(f)
    
    print(f"✅ Caricati {len(summaries)} summary files")
    print(f"   Code indices: {sorted(summaries.keys())}")
    
    return summaries


def extract_metrics_arrays(summaries):
    """
    Estrae le metriche da tutti i summary in array numpy
    
    Returns:
        dict con array per ogni metrica
    """
    code_indices = sorted(summaries.keys())
    
    data = {
        'code_indices': np.array(code_indices),
        'temporal_corr_mean': [],
        'temporal_corr_std': [],
        'spectral_corr_mean': [],
        'spectral_corr_std': [],
        'cosine_dist_mean': [],
        'cosine_dist_std': [],
        'euclidean_dist_mean': [],
        'euclidean_dist_std': [],
    }
    
    for code_idx in code_indices:
        summary = summaries[code_idx]
        stats = summary['statistics']
        
        # Temporal correlation
        data['temporal_corr_mean'].append(stats['temporal_correlation']['mean'])
        data['temporal_corr_std'].append(stats['temporal_correlation']['std'])
        
        # Spectral correlation
        data['spectral_corr_mean'].append(stats['spectral_correlation']['mean'])
        data['spectral_corr_std'].append(stats['spectral_correlation']['std'])
        
        # Cosine distance
        data['cosine_dist_mean'].append(stats['cosine_distance']['mean'])
        data['cosine_dist_std'].append(stats['cosine_distance']['std'])
        
        # Euclidean distance
        data['euclidean_dist_mean'].append(stats['euclidean_distance']['mean'])
        data['euclidean_dist_std'].append(stats['euclidean_distance']['std'])
    
    # Converti liste in array
    for key in data:
        if key != 'code_indices':
            data[key] = np.array(data[key])
    
    return data

# ============================================================================
# FUNZIONI - VISUALIZZAZIONE
# ============================================================================

def plot_scatter_distance_vs_correlation(data, use_wandb=False):
    """
    Crea scatter plot che mostrano la relazione tra distanze e correlazioni
    e li logga su W&B (senza salvataggio locale)
    
    Layout: 2x2 grid
    - Top-Left: Euclidean Distance (x) vs Temporal Correlation (y)
    - Top-Right: Euclidean Distance (x) vs Spectral Correlation (y)
    - Bottom-Left: Cosine Distance (x) vs Temporal Correlation (y)
    - Bottom-Right: Cosine Distance (x) vs Spectral Correlation (y)
    """
    
    codes = data['code_indices']
    
    # Dizionario per salvare le correlazioni
    correlations = {}
    
    # ========================================================================
    # PLOT PRINCIPALE: 2x2 Grid - Scatter Plots
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # ========================================================================
    # PLOT 1: Euclidean Distance vs Temporal Correlation
    # ========================================================================
    ax = axes[0, 0]
    
    x = data['euclidean_dist_mean']
    y = data['temporal_corr_mean']
    
    # Scatter plot con colormap basato sull'indice del codice
    scatter = ax.scatter(x, y, c=codes, cmap='viridis', s=200, 
                        alpha=0.7, edgecolors='black', linewidths=1.5, zorder=3)
    
    # Aggiungi labels per ogni punto
    for i, code in enumerate(codes):
        ax.annotate(f'{code}', (x[i], y[i]), fontsize=9, 
                   ha='center', va='center', color='white', weight='bold')
    
    # Regressione lineare
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, poly(x_line), 'r--', linewidth=2.5, alpha=0.8, label='Linear Fit')
    
    # Calcola correlazione di Pearson
    corr, pval = pearsonr(x, y)
    correlations['euclidean_vs_temporal'] = {'r': corr, 'p': pval, 'slope': coeffs[0]}
    
    ax.set_xlabel('Euclidean Distance (mean)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Temporal Correlation (mean)', fontsize=14, fontweight='bold')
    ax.set_title('Euclidean Distance vs Temporal Correlation', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    # Stats box
    stats_text = f'Pearson r: {corr:.3f}\n'
    stats_text += f'p-value: {pval:.4f}\n'
    stats_text += f'Slope: {coeffs[0]:.3f}\n'
    stats_text += f'n points: {len(codes)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.5))
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Base Code Index', fontsize=12, fontweight='bold')
    
    # ========================================================================
    # PLOT 2: Euclidean Distance vs Spectral Correlation
    # ========================================================================
    ax = axes[0, 1]
    
    x = data['euclidean_dist_mean']
    y = data['spectral_corr_mean']
    
    scatter = ax.scatter(x, y, c=codes, cmap='viridis', s=200, 
                        alpha=0.7, edgecolors='black', linewidths=1.5, zorder=3)
    
    for i, code in enumerate(codes):
        ax.annotate(f'{code}', (x[i], y[i]), fontsize=9, 
                   ha='center', va='center', color='white', weight='bold')
    
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, poly(x_line), 'r--', linewidth=2.5, alpha=0.8, label='Linear Fit')
    
    corr, pval = pearsonr(x, y)
    correlations['euclidean_vs_spectral'] = {'r': corr, 'p': pval, 'slope': coeffs[0]}
    
    ax.set_xlabel('Euclidean Distance (mean)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spectral Correlation (mean)', fontsize=14, fontweight='bold')
    ax.set_title('Euclidean Distance vs Spectral Correlation', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    stats_text = f'Pearson r: {corr:.3f}\n'
    stats_text += f'p-value: {pval:.4f}\n'
    stats_text += f'Slope: {coeffs[0]:.3f}\n'
    stats_text += f'n points: {len(codes)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.5))
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Base Code Index', fontsize=12, fontweight='bold')
    
    # ========================================================================
    # PLOT 3: Cosine Distance vs Temporal Correlation
    # ========================================================================
    ax = axes[1, 0]
    
    x = data['cosine_dist_mean']
    y = data['temporal_corr_mean']
    
    scatter = ax.scatter(x, y, c=codes, cmap='viridis', s=200, 
                        alpha=0.7, edgecolors='black', linewidths=1.5, zorder=3)
    
    for i, code in enumerate(codes):
        ax.annotate(f'{code}', (x[i], y[i]), fontsize=9, 
                   ha='center', va='center', color='white', weight='bold')
    
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, poly(x_line), 'r--', linewidth=2.5, alpha=0.8, label='Linear Fit')
    
    corr, pval = pearsonr(x, y)
    correlations['cosine_vs_temporal'] = {'r': corr, 'p': pval, 'slope': coeffs[0]}
    
    ax.set_xlabel('Cosine Distance (mean)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Temporal Correlation (mean)', fontsize=14, fontweight='bold')
    ax.set_title('Cosine Distance vs Temporal Correlation', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    stats_text = f'Pearson r: {corr:.3f}\n'
    stats_text += f'p-value: {pval:.4f}\n'
    stats_text += f'Slope: {coeffs[0]:.3f}\n'
    stats_text += f'n points: {len(codes)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.5))
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Base Code Index', fontsize=12, fontweight='bold')
    
    # ========================================================================
    # PLOT 4: Cosine Distance vs Spectral Correlation
    # ========================================================================
    ax = axes[1, 1]
    
    x = data['cosine_dist_mean']
    y = data['spectral_corr_mean']
    
    scatter = ax.scatter(x, y, c=codes, cmap='viridis', s=200, 
                        alpha=0.7, edgecolors='black', linewidths=1.5, zorder=3)
    
    for i, code in enumerate(codes):
        ax.annotate(f'{code}', (x[i], y[i]), fontsize=9, 
                   ha='center', va='center', color='white', weight='bold')
    
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, poly(x_line), 'r--', linewidth=2.5, alpha=0.8, label='Linear Fit')
    
    corr, pval = pearsonr(x, y)
    correlations['cosine_vs_spectral'] = {'r': corr, 'p': pval, 'slope': coeffs[0]}
    
    ax.set_xlabel('Cosine Distance (mean)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spectral Correlation (mean)', fontsize=14, fontweight='bold')
    ax.set_title('Cosine Distance vs Spectral Correlation', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    stats_text = f'Pearson r: {corr:.3f}\n'
    stats_text += f'p-value: {pval:.4f}\n'
    stats_text += f'Slope: {coeffs[0]:.3f}\n'
    stats_text += f'n points: {len(codes)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.5))
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Base Code Index', fontsize=12, fontweight='bold')
    
    # ========================================================================
    # Finalizzazione
    # ========================================================================
    plt.suptitle('Distance-Correlation Relationships Across All Base Codes', 
                fontsize=17, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # ========================================================================
    # Log su W&B (no salvataggio locale)
    # ========================================================================
    if use_wandb:
        # Log l'immagine direttamente dalla figura matplotlib
        wandb.log({
            "distance_correlation_scatter": wandb.Image(fig)
        })
        
        # Log le metriche di correlazione
        for key, vals in correlations.items():
            wandb.log({
                f"correlation/{key}/pearson_r": vals['r'],
                f"correlation/{key}/p_value": vals['p'],
                f"correlation/{key}/slope": vals['slope']
            })
        
        print(f"✅ Grafici e metriche loggati su W&B")
    else:
        print(f"⚠️  W&B non abilitato - nessun output salvato")
    
    plt.close()
    
    return correlations


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("📊 SCATTER PLOT: DISTANZE vs CORRELAZIONI")
    print("="*70)
    print(f"\n📂 Directory input: {SUMMARY_DIR}/")
    print(f"📤 Output: W&B only (no local files)")
    
    try:
        # 0. Inizializza W&B se richiesto
        if USE_WANDB:
            print(f"\n{'='*70}")
            print("🔄 INIZIALIZZAZIONE WEIGHTS & BIASES")
            print(f"{'='*70}")
            wandb.init(
                project=WANDB_PROJECT,
                name=WANDB_RUN_NAME,
                config={
                    "analysis_type": "distance_correlation_scatter",
                    "summary_dir": SUMMARY_DIR,
                }
            )
            print(f"✅ W&B inizializzato - Project: {WANDB_PROJECT}, Run: {WANDB_RUN_NAME}")
        
        # 1. Carica tutti i summary
        print(f"\n{'='*70}")
        print("📥 CARICAMENTO DATI")
        print(f"{'='*70}")
        
        summaries = load_all_summaries(SUMMARY_DIR)
        
        # 2. Estrai metriche in array
        print(f"\n📊 Estrazione metriche...")
        data = extract_metrics_arrays(summaries)
        
        # 3. Crea scatter plot
        print(f"\n{'='*70}")
        print("📈 CREAZIONE SCATTER PLOTS")
        print(f"{'='*70}")
        
        correlations = plot_scatter_distance_vs_correlation(data, use_wandb=USE_WANDB)
        
        # 4. Stampa statistiche sulle correlazioni
        print(f"\n{'='*70}")
        print("📊 ANALISI RELAZIONI DISTANZA-CORRELAZIONE")
        print(f"{'='*70}")
        
        for key, vals in correlations.items():
            metric_name = key.replace('_', ' ').title()
            print(f"\n📈 {metric_name}:")
            print(f"   Pearson r: {vals['r']:.4f} (p={vals['p']:.4f})")
            print(f"   Slope: {vals['slope']:.4f}")
        
        print(f"\n{'='*70}")
        print("✅ ANALISI COMPLETATA!")
        print(f"{'='*70}")
        
        if USE_WANDB:
            print(f"🌐 Visualizza su W&B: {wandb.run.url}")
            wandb.finish()
        else:
            print(f"⚠️  W&B non abilitato - nessun output salvato")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        if USE_WANDB:
            wandb.finish()


if __name__ == "__main__":
    main()