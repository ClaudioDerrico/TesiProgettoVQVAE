#!/usr/bin/env python3
"""
Visualizzazione Scatter: TUTTE le Coppie di Codici (1024 punti)

Invece di plottare medie per codice base, plottiamo OGNI SINGOLA COPPIA:
- (0,0), (0,1), ..., (0,31)
- (1,0), (1,1), ..., (1,31)
- ...
- (31,0), (31,1), ..., (31,31)

Totale: 32 × 32 = 1024 punti per grafico

4 Scatter Plots:
1. Euclidean Distance vs Temporal Correlation
2. Euclidean Distance vs Spectral Correlation
3. Cosine Distance vs Temporal Correlation
4. Cosine Distance vs Spectral Correlation

Uso:
    python plot_all_pairs_scatter.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import wandb

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

PAIRS_DIR = "correlazione_distanze_temporali_spettrali/coppie codice"

# Configurazione W&B
WANDB_PROJECT = "correlazione_vs_distanza_completi"
WANDB_RUN_NAME = "all-pairs-scatter-1024pts"
USE_WANDB = True

# ============================================================================
# CARICAMENTO DATI
# ============================================================================

def load_all_pair_files(pairs_dir):
    """
    Carica tutti i file coppie_codice_*.json
    
    Returns:
        dict: {base_code: pairs_data}
    """
    pairs_dir_path = Path(pairs_dir)
    
    if not pairs_dir_path.exists():
        raise FileNotFoundError(f"Directory {pairs_dir} non trovata!")
    
    # Pattern: coppie_codice_*.json
    pair_files = sorted(pairs_dir_path.glob("coppie_codice_*.json"))
    
    if not pair_files:
        # Prova senza underscore
        pair_files = sorted(pairs_dir_path.glob("coppiecodice*.json"))
    
    if not pair_files:
        raise FileNotFoundError(f"Nessun file coppie_codice_*.json trovato in {pairs_dir}/")
    
    all_pairs_data = {}
    
    for file_path in pair_files:
        # Estrai base code dal nome file
        filename = file_path.stem
        
        try:
            # Cerca il numero finale
            import re
            match = re.search(r'(\d+)$', filename)
            if match:
                base_code = int(match.group(1))
            else:
                print(f"⚠️  Skipping {file_path.name}: cannot extract code index")
                continue
        except Exception as e:
            print(f"⚠️  Skipping {file_path.name}: {e}")
            continue
        
        # Carica il file
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_pairs_data[base_code] = data
    
    print(f"✅ Caricati {len(all_pairs_data)} file di coppie")
    print(f"   Base codes: {sorted(all_pairs_data.keys())}")
    
    return all_pairs_data


def extract_all_pairs_data(all_pairs_data):
    """
    Estrae TUTTE le singole coppie in array flat
    
    Returns:
        dict con array per ogni metrica (lunghezza = num_codes²)
    """
    
    all_base_codes = []
    all_target_codes = []
    all_cosine_dist = []
    all_euclidean_dist = []
    all_temporal_corr = []
    all_spectral_corr = []
    
    # Itera su ogni base code
    for base_code in sorted(all_pairs_data.keys()):
        pairs_list = all_pairs_data[base_code]['pairs']
        
        # Per ogni coppia in questo file
        for pair in pairs_list:
            code_pair = pair['code_pair']
            
            all_base_codes.append(code_pair[0])
            all_target_codes.append(code_pair[1])
            all_cosine_dist.append(pair['cosine_distance'])
            all_euclidean_dist.append(pair['euclidean_distance'])
            all_temporal_corr.append(pair['temporal_correlation'])
            all_spectral_corr.append(pair['spectral_correlation'])
    
    data = {
        'base_codes': np.array(all_base_codes),
        'target_codes': np.array(all_target_codes),
        'cosine_distance': np.array(all_cosine_dist),
        'euclidean_distance': np.array(all_euclidean_dist),
        'temporal_correlation': np.array(all_temporal_corr),
        'spectral_correlation': np.array(all_spectral_corr),
    }
    
    print(f"\n✅ Estratte {len(data['base_codes'])} coppie totali")
    print(f"   Range base codes: {data['base_codes'].min()} - {data['base_codes'].max()}")
    print(f"   Range target codes: {data['target_codes'].min()} - {data['target_codes'].max()}")
    
    return data


# ============================================================================
# VISUALIZZAZIONE
# ============================================================================

def plot_all_pairs_scatter(data, use_wandb=False):
    """
    Crea 4 scatter plots con TUTTE le 1024 coppie
    
    Colori: gradiente basato su base_code (quale codice è il riferimento)
    """
    
    # Per colorare i punti in base al base_code
    base_codes = data['base_codes']
    
    # Dizionario per salvare le correlazioni
    correlations = {}
    
    # ========================================================================
    # PLOT PRINCIPALE: 2x2 Grid
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    
    # ========================================================================
    # PLOT 1: Euclidean Distance vs Temporal Correlation
    # ========================================================================
    ax = axes[0, 0]
    
    x = data['euclidean_distance']
    y = data['temporal_correlation']
    
    # Scatter con colormap basato su base_code
    scatter = ax.scatter(x, y, c=base_codes, cmap='viridis', 
                        s=30, alpha=0.6, edgecolors='none')
    
    # Regressione lineare
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, poly(x_line), 'r--', linewidth=3, alpha=0.8, 
           label=f'Linear Fit (slope={coeffs[0]:.3f})')
    
    # Correlazioni
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    
    correlations['euclidean_vs_temporal'] = {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'slope': coeffs[0]
    }
    
    ax.set_xlabel('Euclidean Distance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Temporal Correlation', fontsize=14, fontweight='bold')
    ax.set_title('Euclidean Distance vs Temporal Correlation\n(All 1024 pairs)', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    # Stats box
    stats_text = f'Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})\n'
    stats_text += f'Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})\n'
    stats_text += f'Slope: {coeffs[0]:.4f}\n'
    stats_text += f'N pairs: {len(x)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, 
                    edgecolor='black', linewidth=1.5))
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Base Code Index', fontsize=12, fontweight='bold')
    
    # ========================================================================
    # PLOT 2: Euclidean Distance vs Spectral Correlation
    # ========================================================================
    ax = axes[0, 1]
    
    x = data['euclidean_distance']
    y = data['spectral_correlation']
    
    scatter = ax.scatter(x, y, c=base_codes, cmap='plasma', 
                        s=30, alpha=0.6, edgecolors='none')
    
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, poly(x_line), 'r--', linewidth=3, alpha=0.8,
           label=f'Linear Fit (slope={coeffs[0]:.3f})')
    
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    
    correlations['euclidean_vs_spectral'] = {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'slope': coeffs[0]
    }
    
    ax.set_xlabel('Euclidean Distance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spectral Correlation', fontsize=14, fontweight='bold')
    ax.set_title('Euclidean Distance vs Spectral Correlation\n(All 1024 pairs)', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    stats_text = f'Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})\n'
    stats_text += f'Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})\n'
    stats_text += f'Slope: {coeffs[0]:.4f}\n'
    stats_text += f'N pairs: {len(x)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                    edgecolor='black', linewidth=1.5))
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Base Code Index', fontsize=12, fontweight='bold')
    
    # ========================================================================
    # PLOT 3: Cosine Distance vs Temporal Correlation
    # ========================================================================
    ax = axes[1, 0]
    
    x = data['cosine_distance']
    y = data['temporal_correlation']
    
    scatter = ax.scatter(x, y, c=base_codes, cmap='coolwarm', 
                        s=30, alpha=0.6, edgecolors='none')
    
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, poly(x_line), 'r--', linewidth=3, alpha=0.8,
           label=f'Linear Fit (slope={coeffs[0]:.3f})')
    
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    
    correlations['cosine_vs_temporal'] = {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'slope': coeffs[0]
    }
    
    ax.set_xlabel('Cosine Distance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Temporal Correlation', fontsize=14, fontweight='bold')
    ax.set_title('Cosine Distance vs Temporal Correlation\n(All 1024 pairs)', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    stats_text = f'Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})\n'
    stats_text += f'Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})\n'
    stats_text += f'Slope: {coeffs[0]:.4f}\n'
    stats_text += f'N pairs: {len(x)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                    edgecolor='black', linewidth=1.5))
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Base Code Index', fontsize=12, fontweight='bold')
    
    # ========================================================================
    # PLOT 4: Cosine Distance vs Spectral Correlation
    # ========================================================================
    ax = axes[1, 1]
    
    x = data['cosine_distance']
    y = data['spectral_correlation']
    
    scatter = ax.scatter(x, y, c=base_codes, cmap='RdYlGn', 
                        s=30, alpha=0.6, edgecolors='none')
    
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, poly(x_line), 'r--', linewidth=3, alpha=0.8,
           label=f'Linear Fit (slope={coeffs[0]:.3f})')
    
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    
    correlations['cosine_vs_spectral'] = {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'slope': coeffs[0]
    }
    
    ax.set_xlabel('Cosine Distance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spectral Correlation', fontsize=14, fontweight='bold')
    ax.set_title('Cosine Distance vs Spectral Correlation\n(All 1024 pairs)', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    stats_text = f'Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})\n'
    stats_text += f'Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})\n'
    stats_text += f'Slope: {coeffs[0]:.4f}\n'
    stats_text += f'N pairs: {len(x)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                    edgecolor='black', linewidth=1.5))
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Base Code Index', fontsize=12, fontweight='bold')
    
    # ========================================================================
    # Finalizzazione
    # ========================================================================
    plt.suptitle('Distance-Correlation Relationships: All Code Pairs (1024 points)', 
                fontsize=17, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # ========================================================================
    # Log su W&B
    # ========================================================================
    if use_wandb:
        wandb.log({
            "all_pairs_scatter": wandb.Image(fig)
        })
        
        # Log metriche
        for key, vals in correlations.items():
            wandb.log({
                f"correlation/{key}/pearson_r": vals['pearson_r'],
                f"correlation/{key}/pearson_p": vals['pearson_p'],
                f"correlation/{key}/spearman_r": vals['spearman_r'],
                f"correlation/{key}/spearman_p": vals['spearman_p'],
                f"correlation/{key}/slope": vals['slope']
            })
        
        print(f"✅ Grafico e metriche loggati su W&B")
    
    plt.close()
    
    return correlations


# ============================================================================
# ANALISI STATISTICA
# ============================================================================

def print_detailed_statistics(data, correlations):
    """Stampa statistiche dettagliate"""
    
    print(f"\n{'='*70}")
    print("📊 STATISTICHE DESCRITTIVE")
    print(f"{'='*70}")
    
    print(f"\n🔵 TEMPORAL CORRELATION:")
    print(f"   Mean: {data['temporal_correlation'].mean():.4f}")
    print(f"   Std:  {data['temporal_correlation'].std():.4f}")
    print(f"   Min:  {data['temporal_correlation'].min():.4f}")
    print(f"   Max:  {data['temporal_correlation'].max():.4f}")
    print(f"   Positive: {np.sum(data['temporal_correlation'] > 0)} pairs ({100*np.sum(data['temporal_correlation'] > 0)/len(data['temporal_correlation']):.1f}%)")
    print(f"   Negative: {np.sum(data['temporal_correlation'] < 0)} pairs ({100*np.sum(data['temporal_correlation'] < 0)/len(data['temporal_correlation']):.1f}%)")
    
    print(f"\n🟣 SPECTRAL CORRELATION:")
    print(f"   Mean: {data['spectral_correlation'].mean():.4f}")
    print(f"   Std:  {data['spectral_correlation'].std():.4f}")
    print(f"   Min:  {data['spectral_correlation'].min():.4f}")
    print(f"   Max:  {data['spectral_correlation'].max():.4f}")
    print(f"   High (>0.9): {np.sum(data['spectral_correlation'] > 0.9)} pairs ({100*np.sum(data['spectral_correlation'] > 0.9)/len(data['spectral_correlation']):.1f}%)")
    
    print(f"\n🟢 COSINE DISTANCE:")
    print(f"   Mean: {data['cosine_distance'].mean():.4f}")
    print(f"   Std:  {data['cosine_distance'].std():.4f}")
    print(f"   Min:  {data['cosine_distance'].min():.4f}")
    print(f"   Max:  {data['cosine_distance'].max():.4f}")
    
    print(f"\n🟡 EUCLIDEAN DISTANCE:")
    print(f"   Mean: {data['euclidean_distance'].mean():.4f}")
    print(f"   Std:  {data['euclidean_distance'].std():.4f}")
    print(f"   Min:  {data['euclidean_distance'].min():.4f}")
    print(f"   Max:  {data['euclidean_distance'].max():.4f}")
    
    print(f"\n{'='*70}")
    print("📈 RELAZIONI DISTANZA-CORRELAZIONE")
    print(f"{'='*70}")
    
    for key, vals in correlations.items():
        metric_name = key.replace('_', ' ').title()
        print(f"\n{metric_name}:")
        print(f"   Pearson r:  {vals['pearson_r']:.4f} (p={vals['pearson_p']:.2e})")
        print(f"   Spearman ρ: {vals['spearman_r']:.4f} (p={vals['spearman_p']:.2e})")
        print(f"   Slope:      {vals['slope']:.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("📊 SCATTER PLOT: TUTTE LE COPPIE (1024 punti)")
    print("="*70)
    print(f"\n📂 Directory input: {PAIRS_DIR}/")
    print(f"📤 Output: W&B only")
    
    try:
        # Inizializza W&B
        if USE_WANDB:
            print(f"\n{'='*70}")
            print("🔄 INIZIALIZZAZIONE W&B")
            print(f"{'='*70}")
            wandb.init(
                project=WANDB_PROJECT,
                name=WANDB_RUN_NAME,
                config={
                    "analysis_type": "all_pairs_scatter",
                    "pairs_dir": PAIRS_DIR,
                    "total_pairs": 1024,
                }
            )
            print(f"✅ W&B inizializzato")
        
        # 1. Carica tutti i file
        print(f"\n{'='*70}")
        print("📥 CARICAMENTO DATI")
        print(f"{'='*70}")
        
        all_pairs_data = load_all_pair_files(PAIRS_DIR)
        
        # 2. Estrai tutte le coppie
        print(f"\n📊 Estrazione coppie...")
        data = extract_all_pairs_data(all_pairs_data)
        
        # 3. Crea scatter plot
        print(f"\n{'='*70}")
        print("📈 CREAZIONE SCATTER PLOTS")
        print(f"{'='*70}")
        
        correlations = plot_all_pairs_scatter(data, use_wandb=USE_WANDB)
        
        # 4. Statistiche
        print_detailed_statistics(data, correlations)
        
        # 5. Log metriche descrittive
        if USE_WANDB:
            wandb.log({
                'descriptive/temporal_corr_mean': data['temporal_correlation'].mean(),
                'descriptive/temporal_corr_std': data['temporal_correlation'].std(),
                'descriptive/spectral_corr_mean': data['spectral_correlation'].mean(),
                'descriptive/spectral_corr_std': data['spectral_correlation'].std(),
                'descriptive/cosine_dist_mean': data['cosine_distance'].mean(),
                'descriptive/cosine_dist_std': data['cosine_distance'].std(),
                'descriptive/euclidean_dist_mean': data['euclidean_distance'].mean(),
                'descriptive/euclidean_dist_std': data['euclidean_distance'].std(),
            })
        
        print(f"\n{'='*70}")
        print("✅ ANALISI COMPLETATA!")
        print(f"{'='*70}")
        
        if USE_WANDB:
            print(f"🌐 W&B Dashboard: {wandb.run.url}")
            wandb.finish()
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        if USE_WANDB:
            wandb.finish()


if __name__ == "__main__":
    main()