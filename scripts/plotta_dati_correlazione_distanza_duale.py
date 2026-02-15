#!/usr/bin/env python3
"""
Visualizzazione Scatter: TUTTE le Coppie di Codici DUAL (1024 punti)

Versione DUAL che separa le coppie per tipo di codebook:
- Active-Active
- Active-Inactive
- Inactive-Active
- Inactive-Inactive

4 Scatter Plots con colori diversi per tipo di coppia:
1. Euclidean Distance vs Temporal Correlation
2. Euclidean Distance vs Spectral Correlation
3. Cosine Distance vs Temporal Correlation
4. Cosine Distance vs Spectral Correlation

Uso:
    python plotta_dati_correlazione_distanza_duale.py
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

PAIRS_DIR = "correlazione_distanze_temporali_spettrali_dual/coppie codice"

# Configurazione W&B
WANDB_PROJECT = "correlazione_vs_distanza_duale_completi"
WANDB_RUN_NAME = "all-pairs-scatter-dual-1024pts"
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
    Estrae TUTTE le singole coppie in array flat con informazioni dual
    
    Returns:
        dict con array per ogni metrica (lunghezza = num_codes²)
        + array per tipo di coppia
    """
    
    all_base_codes = []
    all_target_codes = []
    all_cosine_dist = []
    all_euclidean_dist = []
    all_temporal_corr = []
    all_spectral_corr = []
    all_pair_types = []  # "active-active", "active-inactive", etc.
    
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
            
            # Estrai tipo di coppia
            base_type = pair.get('base_codebook_type', 'unknown')
            target_type = pair.get('target_codebook_type', 'unknown')
            pair_type = f"{base_type}-{target_type}"
            all_pair_types.append(pair_type)
    
    data = {
        'base_codes': np.array(all_base_codes),
        'target_codes': np.array(all_target_codes),
        'cosine_distance': np.array(all_cosine_dist),
        'euclidean_distance': np.array(all_euclidean_dist),
        'temporal_correlation': np.array(all_temporal_corr),
        'spectral_correlation': np.array(all_spectral_corr),
        'pair_types': np.array(all_pair_types),
    }
    
    print(f"\n✅ Estratte {len(data['base_codes'])} coppie totali")
    print(f"   Range base codes: {data['base_codes'].min()} - {data['base_codes'].max()}")
    print(f"   Range target codes: {data['target_codes'].min()} - {data['target_codes'].max()}")
    
    # Statistiche per tipo
    unique_types, counts = np.unique(data['pair_types'], return_counts=True)
    print(f"\n📊 Distribuzione per tipo di coppia:")
    for pair_type, count in zip(unique_types, counts):
        print(f"   {pair_type}: {count} coppie ({100*count/len(data['pair_types']):.1f}%)")
    
    return data


# ============================================================================
# VISUALIZZAZIONE
# ============================================================================

def plot_all_pairs_scatter_dual(data, use_wandb=False):
    """
    Crea 4 scatter plots con TUTTE le 1024 coppie
    Colori diversi per tipo di coppia (active-active, active-inactive, etc.)
    """
    
    # Colori per tipo di coppia
    pair_type_colors = {
        'active-active': 'blue',
        'active-inactive': 'orange',
        'inactive-active': 'green',
        'inactive-inactive': 'red',
    }
    
    pair_type_labels = {
        'active-active': 'Active-Active',
        'active-inactive': 'Active-Inactive',
        'inactive-active': 'Inactive-Active',
        'inactive-inactive': 'Inactive-Inactive',
    }
    
    # Dizionario per salvare le correlazioni (per tipo e globali)
    correlations = {}
    correlations_by_type = {}
    
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
    pair_types = data['pair_types']
    
    # Plot per ogni tipo di coppia
    for pair_type in pair_type_colors.keys():
        mask = pair_types == pair_type
        if np.any(mask):
            ax.scatter(x[mask], y[mask], 
                      c=pair_type_colors[pair_type], 
                      label=pair_type_labels[pair_type],
                      s=30, alpha=0.6, edgecolors='none')
    
    # Regressione lineare globale
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) > 1:
        coeffs = np.polyfit(x[valid_mask], y[valid_mask], 1)
        poly = np.poly1d(coeffs)
        x_line = np.linspace(x[valid_mask].min(), x[valid_mask].max(), 100)
        ax.plot(x_line, poly(x_line), 'k--', linewidth=3, alpha=0.8, 
               label=f'Global Fit (slope={coeffs[0]:.3f})')
    
    # Correlazioni globali
    if np.sum(valid_mask) > 1:
        pearson_r, pearson_p = pearsonr(x[valid_mask], y[valid_mask])
        spearman_r, spearman_p = spearmanr(x[valid_mask], y[valid_mask])
        
        correlations['euclidean_vs_temporal'] = {
            'pearson_r': pearson_r, 'pearson_p': pearson_p,
            'spearman_r': spearman_r, 'spearman_p': spearman_p,
            'slope': coeffs[0] if np.sum(valid_mask) > 1 else np.nan
        }
        
        # Correlazioni per tipo
        for pair_type in pair_type_colors.keys():
            mask = (pair_types == pair_type) & valid_mask
            if np.sum(mask) > 1:
                pearson_r_t, pearson_p_t = pearsonr(x[mask], y[mask])
                spearman_r_t, spearman_p_t = spearmanr(x[mask], y[mask])
                correlations_by_type[f'euclidean_vs_temporal_{pair_type}'] = {
                    'pearson_r': pearson_r_t, 'pearson_p': pearson_p_t,
                    'spearman_r': spearman_r_t, 'spearman_p': spearman_p_t,
                }
    
    ax.set_xlabel('Euclidean Distance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Temporal Correlation', fontsize=14, fontweight='bold')
    ax.set_title('Euclidean Distance vs Temporal Correlation\n(All 1024 pairs, DUAL)', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    # Stats box
    if np.sum(valid_mask) > 1:
        stats_text = f'Global:\n'
        stats_text += f'Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})\n'
        stats_text += f'Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})\n'
        stats_text += f'Slope: {coeffs[0]:.4f}\n'
        stats_text += f'N pairs: {np.sum(valid_mask)}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, 
                        edgecolor='black', linewidth=1.5))
    
    # ========================================================================
    # PLOT 2: Euclidean Distance vs Spectral Correlation
    # ========================================================================
    ax = axes[0, 1]
    
    x = data['euclidean_distance']
    y = data['spectral_correlation']
    
    for pair_type in pair_type_colors.keys():
        mask = pair_types == pair_type
        if np.any(mask):
            ax.scatter(x[mask], y[mask], 
                      c=pair_type_colors[pair_type], 
                      label=pair_type_labels[pair_type],
                      s=30, alpha=0.6, edgecolors='none')
    
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) > 1:
        coeffs = np.polyfit(x[valid_mask], y[valid_mask], 1)
        poly = np.poly1d(coeffs)
        x_line = np.linspace(x[valid_mask].min(), x[valid_mask].max(), 100)
        ax.plot(x_line, poly(x_line), 'k--', linewidth=3, alpha=0.8,
               label=f'Global Fit (slope={coeffs[0]:.3f})')
        
        pearson_r, pearson_p = pearsonr(x[valid_mask], y[valid_mask])
        spearman_r, spearman_p = spearmanr(x[valid_mask], y[valid_mask])
        
        correlations['euclidean_vs_spectral'] = {
            'pearson_r': pearson_r, 'pearson_p': pearson_p,
            'spearman_r': spearman_r, 'spearman_p': spearman_p,
            'slope': coeffs[0]
        }
        
        # Per tipo
        for pair_type in pair_type_colors.keys():
            mask = (pair_types == pair_type) & valid_mask
            if np.sum(mask) > 1:
                pearson_r_t, pearson_p_t = pearsonr(x[mask], y[mask])
                spearman_r_t, spearman_p_t = spearmanr(x[mask], y[mask])
                correlations_by_type[f'euclidean_vs_spectral_{pair_type}'] = {
                    'pearson_r': pearson_r_t, 'pearson_p': pearson_p_t,
                    'spearman_r': spearman_r_t, 'spearman_p': spearman_p_t,
                }
    
    ax.set_xlabel('Euclidean Distance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spectral Correlation', fontsize=14, fontweight='bold')
    ax.set_title('Euclidean Distance vs Spectral Correlation\n(All 1024 pairs, DUAL)', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    if np.sum(valid_mask) > 1:
        stats_text = f'Global:\n'
        stats_text += f'Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})\n'
        stats_text += f'Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})\n'
        stats_text += f'Slope: {coeffs[0]:.4f}\n'
        stats_text += f'N pairs: {np.sum(valid_mask)}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                        edgecolor='black', linewidth=1.5))
    
    # ========================================================================
    # PLOT 3: Cosine Distance vs Temporal Correlation
    # ========================================================================
    ax = axes[1, 0]
    
    x = data['cosine_distance']
    y = data['temporal_correlation']
    
    for pair_type in pair_type_colors.keys():
        mask = pair_types == pair_type
        if np.any(mask):
            ax.scatter(x[mask], y[mask], 
                      c=pair_type_colors[pair_type], 
                      label=pair_type_labels[pair_type],
                      s=30, alpha=0.6, edgecolors='none')
    
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) > 1:
        coeffs = np.polyfit(x[valid_mask], y[valid_mask], 1)
        poly = np.poly1d(coeffs)
        x_line = np.linspace(x[valid_mask].min(), x[valid_mask].max(), 100)
        ax.plot(x_line, poly(x_line), 'k--', linewidth=3, alpha=0.8,
               label=f'Global Fit (slope={coeffs[0]:.3f})')
        
        pearson_r, pearson_p = pearsonr(x[valid_mask], y[valid_mask])
        spearman_r, spearman_p = spearmanr(x[valid_mask], y[valid_mask])
        
        correlations['cosine_vs_temporal'] = {
            'pearson_r': pearson_r, 'pearson_p': pearson_p,
            'spearman_r': spearman_r, 'spearman_p': spearman_p,
            'slope': coeffs[0]
        }
        
        # Per tipo
        for pair_type in pair_type_colors.keys():
            mask = (pair_types == pair_type) & valid_mask
            if np.sum(mask) > 1:
                pearson_r_t, pearson_p_t = pearsonr(x[mask], y[mask])
                spearman_r_t, spearman_p_t = spearmanr(x[mask], y[mask])
                correlations_by_type[f'cosine_vs_temporal_{pair_type}'] = {
                    'pearson_r': pearson_r_t, 'pearson_p': pearson_p_t,
                    'spearman_r': spearman_r_t, 'spearman_p': spearman_p_t,
                }
    
    ax.set_xlabel('Cosine Distance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Temporal Correlation', fontsize=14, fontweight='bold')
    ax.set_title('Cosine Distance vs Temporal Correlation\n(All 1024 pairs, DUAL)', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    if np.sum(valid_mask) > 1:
        stats_text = f'Global:\n'
        stats_text += f'Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})\n'
        stats_text += f'Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})\n'
        stats_text += f'Slope: {coeffs[0]:.4f}\n'
        stats_text += f'N pairs: {np.sum(valid_mask)}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                        edgecolor='black', linewidth=1.5))
    
    # ========================================================================
    # PLOT 4: Cosine Distance vs Spectral Correlation
    # ========================================================================
    ax = axes[1, 1]
    
    x = data['cosine_distance']
    y = data['spectral_correlation']
    
    for pair_type in pair_type_colors.keys():
        mask = pair_types == pair_type
        if np.any(mask):
            ax.scatter(x[mask], y[mask], 
                      c=pair_type_colors[pair_type], 
                      label=pair_type_labels[pair_type],
                      s=30, alpha=0.6, edgecolors='none')
    
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) > 1:
        coeffs = np.polyfit(x[valid_mask], y[valid_mask], 1)
        poly = np.poly1d(coeffs)
        x_line = np.linspace(x[valid_mask].min(), x[valid_mask].max(), 100)
        ax.plot(x_line, poly(x_line), 'k--', linewidth=3, alpha=0.8,
               label=f'Global Fit (slope={coeffs[0]:.3f})')
        
        pearson_r, pearson_p = pearsonr(x[valid_mask], y[valid_mask])
        spearman_r, spearman_p = spearmanr(x[valid_mask], y[valid_mask])
        
        correlations['cosine_vs_spectral'] = {
            'pearson_r': pearson_r, 'pearson_p': pearson_p,
            'spearman_r': spearman_r, 'spearman_p': spearman_p,
            'slope': coeffs[0]
        }
        
        # Per tipo
        for pair_type in pair_type_colors.keys():
            mask = (pair_types == pair_type) & valid_mask
            if np.sum(mask) > 1:
                pearson_r_t, pearson_p_t = pearsonr(x[mask], y[mask])
                spearman_r_t, spearman_p_t = spearmanr(x[mask], y[mask])
                correlations_by_type[f'cosine_vs_spectral_{pair_type}'] = {
                    'pearson_r': pearson_r_t, 'pearson_p': pearson_p_t,
                    'spearman_r': spearman_r_t, 'spearman_p': spearman_p_t,
                }
    
    ax.set_xlabel('Cosine Distance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spectral Correlation', fontsize=14, fontweight='bold')
    ax.set_title('Cosine Distance vs Spectral Correlation\n(All 1024 pairs, DUAL)', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    
    if np.sum(valid_mask) > 1:
        stats_text = f'Global:\n'
        stats_text += f'Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})\n'
        stats_text += f'Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})\n'
        stats_text += f'Slope: {coeffs[0]:.4f}\n'
        stats_text += f'N pairs: {np.sum(valid_mask)}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                        edgecolor='black', linewidth=1.5))
    
    # ========================================================================
    # Finalizzazione
    # ========================================================================
    plt.suptitle('Distance-Correlation Relationships: All Code Pairs DUAL (1024 points)', 
                fontsize=17, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # ========================================================================
    # Log su W&B
    # ========================================================================
    if use_wandb:
        wandb.log({
            "all_pairs_scatter_dual": wandb.Image(fig)
        })
        
        # Log metriche globali
        for key, vals in correlations.items():
            wandb.log({
                f"correlation/{key}/pearson_r": vals['pearson_r'],
                f"correlation/{key}/pearson_p": vals['pearson_p'],
                f"correlation/{key}/spearman_r": vals['spearman_r'],
                f"correlation/{key}/spearman_p": vals['spearman_p'],
                f"correlation/{key}/slope": vals['slope']
            })
        
        # Log metriche per tipo
        for key, vals in correlations_by_type.items():
            wandb.log({
                f"correlation_by_type/{key}/pearson_r": vals['pearson_r'],
                f"correlation_by_type/{key}/pearson_p": vals['pearson_p'],
                f"correlation_by_type/{key}/spearman_r": vals['spearman_r'],
                f"correlation_by_type/{key}/spearman_p": vals['spearman_p'],
            })
        
        print(f"✅ Grafico e metriche loggati su W&B")
    
    plt.close()
    
    return correlations, correlations_by_type


# ============================================================================
# ANALISI STATISTICA
# ============================================================================

def print_detailed_statistics(data, correlations, correlations_by_type):
    """Stampa statistiche dettagliate"""
    
    print(f"\n{'='*70}")
    print("📊 STATISTICHE DESCRITTIVE")
    print(f"{'='*70}")
    
    print(f"\n🔵 TEMPORAL CORRELATION:")
    print(f"   Mean: {np.nanmean(data['temporal_correlation']):.4f}")
    print(f"   Std:  {np.nanstd(data['temporal_correlation']):.4f}")
    print(f"   Min:  {np.nanmin(data['temporal_correlation']):.4f}")
    print(f"   Max:  {np.nanmax(data['temporal_correlation']):.4f}")
    valid = ~np.isnan(data['temporal_correlation'])
    print(f"   Positive: {np.sum(data['temporal_correlation'][valid] > 0)} pairs ({100*np.sum(data['temporal_correlation'][valid] > 0)/np.sum(valid):.1f}%)")
    print(f"   Negative: {np.sum(data['temporal_correlation'][valid] < 0)} pairs ({100*np.sum(data['temporal_correlation'][valid] < 0)/np.sum(valid):.1f}%)")
    
    print(f"\n🟣 SPECTRAL CORRELATION:")
    print(f"   Mean: {np.nanmean(data['spectral_correlation']):.4f}")
    print(f"   Std:  {np.nanstd(data['spectral_correlation']):.4f}")
    print(f"   Min:  {np.nanmin(data['spectral_correlation']):.4f}")
    print(f"   Max:  {np.nanmax(data['spectral_correlation']):.4f}")
    valid = ~np.isnan(data['spectral_correlation'])
    print(f"   High (>0.9): {np.sum(data['spectral_correlation'][valid] > 0.9)} pairs ({100*np.sum(data['spectral_correlation'][valid] > 0.9)/np.sum(valid):.1f}%)")
    
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
    print("📈 RELAZIONI DISTANZA-CORRELAZIONE (GLOBALI)")
    print(f"{'='*70}")
    
    for key, vals in correlations.items():
        metric_name = key.replace('_', ' ').title()
        print(f"\n{metric_name}:")
        print(f"   Pearson r:  {vals['pearson_r']:.4f} (p={vals['pearson_p']:.2e})")
        print(f"   Spearman ρ: {vals['spearman_r']:.4f} (p={vals['spearman_p']:.2e})")
        print(f"   Slope:      {vals['slope']:.4f}")
    
    print(f"\n{'='*70}")
    print("📈 RELAZIONI DISTANZA-CORRELAZIONE (PER TIPO)")
    print(f"{'='*70}")
    
    for key, vals in correlations_by_type.items():
        metric_name = key.replace('_', ' ').title()
        print(f"\n{metric_name}:")
        print(f"   Pearson r:  {vals['pearson_r']:.4f} (p={vals['pearson_p']:.2e})")
        print(f"   Spearman ρ: {vals['spearman_r']:.4f} (p={vals['spearman_p']:.2e})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("📊 SCATTER PLOT: TUTTE LE COPPIE DUAL (1024 punti)")
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
                    "analysis_type": "all_pairs_scatter_dual",
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
        
        correlations, correlations_by_type = plot_all_pairs_scatter_dual(data, use_wandb=USE_WANDB)
        
        # 4. Statistiche
        print_detailed_statistics(data, correlations, correlations_by_type)
        
        # 5. Log metriche descrittive
        if USE_WANDB:
            wandb.log({
                'descriptive/temporal_corr_mean': np.nanmean(data['temporal_correlation']),
                'descriptive/temporal_corr_std': np.nanstd(data['temporal_correlation']),
                'descriptive/spectral_corr_mean': np.nanmean(data['spectral_correlation']),
                'descriptive/spectral_corr_std': np.nanstd(data['spectral_correlation']),
                'descriptive/cosine_dist_mean': data['cosine_distance'].mean(),
                'descriptive/cosine_dist_std': data['cosine_distance'].std(),
                'descriptive/euclidean_dist_mean': data['euclidean_distance'].mean(),
                'descriptive/euclidean_dist_std': data['euclidean_distance'].std(),
            })
            
            # Statistiche per tipo
            for pair_type in ['active-active', 'active-inactive', 'inactive-active', 'inactive-inactive']:
                mask = data['pair_types'] == pair_type
                if np.any(mask):
                    wandb.log({
                        f'descriptive_by_type/{pair_type}/temporal_corr_mean': np.nanmean(data['temporal_correlation'][mask]),
                        f'descriptive_by_type/{pair_type}/spectral_corr_mean': np.nanmean(data['spectral_correlation'][mask]),
                        f'descriptive_by_type/{pair_type}/cosine_dist_mean': data['cosine_distance'][mask].mean(),
                        f'descriptive_by_type/{pair_type}/euclidean_dist_mean': data['euclidean_distance'][mask].mean(),
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

