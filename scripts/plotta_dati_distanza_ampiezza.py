#!/usr/bin/env python3
"""
Visualizzazione Scatter: Distanza vs Differenza di Ampiezza

Plottiamo OGNI SINGOLA COPPIA analizzando la relazione tra:
- Distanza nello spazio dei codici (Euclidean/Cosine)
- Differenza di ampiezza (magnitude) tra i codici

L'ipotesi è che codici vicini (bassa distanza) dovrebbero avere 
differenze di ampiezza basse.

Totale: 32 × 32 = 1024 punti per grafico

4 Scatter Plots:
1. Euclidean Distance vs Amplitude Difference (Absolute)
2. Euclidean Distance vs Amplitude Difference (Relative %)
3. Cosine Distance vs Amplitude Difference (Absolute)
4. Cosine Distance vs Amplitude Difference (Relative %)

Uso:
    python plotta_dati_distanza_ampiezza.py
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

AMPLITUDE_PAIRS_DIR = "ampiezza_coppie_dual/coppie codice"

# Configurazione W&B
WANDB_PROJECT = "ampiezza_vs_distanza_completi"
WANDB_RUN_NAME = "amplitude-difference-scatter-duale"
USE_WANDB = True

# ============================================================================
# CARICAMENTO DATI
# ============================================================================

def load_all_amplitude_pair_files(pairs_dir):
    """
    Carica tutti i file ampiezza_coppie_codice_*.json
    
    Returns:
        dict: {base_code: pairs_data}
    """
    pairs_dir_path = Path(pairs_dir)
    
    if not pairs_dir_path.exists():
        raise FileNotFoundError(f"Directory {pairs_dir} non trovata!")
    
    # Pattern per i file di ampiezza
    pair_files = sorted(pairs_dir_path.glob("ampiezza_coppie_codice_*.json"))
    
    if not pair_files:
        raise FileNotFoundError(f"Nessun file ampiezza_coppie_codice_*.json trovato in {pairs_dir}/")
    
    all_pairs_data = {}
    
    for file_path in pair_files:
        filename = file_path.stem
        
        try:
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
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_pairs_data[base_code] = data
    
    print(f"✅ Caricati {len(all_pairs_data)} file di ampiezze")
    print(f"   Base codes: {sorted(all_pairs_data.keys())}")
    
    return all_pairs_data


def extract_amplitude_data(all_pairs_data):
    """
    Estrae TUTTE le singole coppie con dati di ampiezza e distanza
    
    Returns:
        dict con array per ogni metrica (lunghezza = num_codes²)
    """
    
    all_base_codes = []
    all_target_codes = []
    all_base_amplitudes = []
    all_target_amplitudes = []
    all_amplitude_diff_abs = []
    all_amplitude_diff_rel = []
    
    # Itera su ogni base code
    for base_code in sorted(all_pairs_data.keys()):
        pairs_list = all_pairs_data[base_code]['pairs']
        
        # Per ogni coppia in questo file
        for pair in pairs_list:
            code_pair = pair['code_pair']
            
            all_base_codes.append(code_pair[0])
            all_target_codes.append(code_pair[1])
            all_base_amplitudes.append(pair['base_amplitude'])
            all_target_amplitudes.append(pair['target_amplitude'])
            all_amplitude_diff_abs.append(pair['amplitude_difference_abs'])
            all_amplitude_diff_rel.append(pair['amplitude_difference_rel'])
    
    data = {
        'base_codes': np.array(all_base_codes),
        'target_codes': np.array(all_target_codes),
        'base_amplitudes': np.array(all_base_amplitudes),
        'target_amplitudes': np.array(all_target_amplitudes),
        'amplitude_diff_abs': np.array(all_amplitude_diff_abs),
        'amplitude_diff_rel': np.array(all_amplitude_diff_rel),
    }
    
    print(f"\n✅ Estratte {len(data['base_codes'])} coppie totali")
    print(f"   Base amplitudes:")
    print(f"   - Mean: {data['base_amplitudes'].mean():.4f}")
    print(f"   - Std:  {data['base_amplitudes'].std():.4f}")
    print(f"   Amplitude diff (absolute):")
    print(f"   - Mean: {data['amplitude_diff_abs'].mean():.4f}")
    print(f"   - Std:  {data['amplitude_diff_abs'].std():.4f}")
    print(f"   - Min:  {data['amplitude_diff_abs'].min():.4f}")
    print(f"   - Max:  {data['amplitude_diff_abs'].max():.4f}")
    print(f"   Amplitude diff (relative %):")
    print(f"   - Mean: {data['amplitude_diff_rel'].mean():.2f}%")
    print(f"   - Std:  {data['amplitude_diff_rel'].std():.2f}%")
    print(f"   - Min:  {data['amplitude_diff_rel'].min():.2f}%")
    print(f"   - Max:  {data['amplitude_diff_rel'].max():.2f}%")
    
    return data


def compute_distances_from_amplitudes(data):
    """
    Calcola distanze euclidee e cosine dagli embeddings di ampiezza
    Per semplicità, usiamo le ampiezze stesse come proxy
    """
    # Per ora, calcoliamo distanze semplici basate sulle ampiezze
    base_amps = data['base_amplitudes']
    target_amps = data['target_amplitudes']
    
    # Euclidean distance approssimata dalla differenza di ampiezza
    euclidean_dist = np.abs(base_amps - target_amps)
    
    # Cosine distance approssimata (normalized)
    # cos(θ) ≈ 1 quando le ampiezze sono simili
    cosine_dist = 1 - (np.minimum(base_amps, target_amps) / np.maximum(base_amps, target_amps))
    
    data['euclidean_distance'] = euclidean_dist
    data['cosine_distance'] = cosine_dist
    
    print(f"\n✅ Distanze calcolate:")
    print(f"   Euclidean distance:")
    print(f"   - Mean: {euclidean_dist.mean():.4f}")
    print(f"   - Std:  {euclidean_dist.std():.4f}")
    print(f"   Cosine distance:")
    print(f"   - Mean: {cosine_dist.mean():.4f}")
    print(f"   - Std:  {cosine_dist.std():.4f}")
    
    return data


# ============================================================================
# VISUALIZZAZIONE
# ============================================================================

def plot_amplitude_difference_scatter(data, use_wandb=False):
    """
    Crea 4 scatter plots: Distanza vs Differenza Ampiezza
    
    Ipotesi: Bassa distanza → Bassa differenza ampiezza
    """
    
    base_codes = data['base_codes']
    correlations = {}
    
    # ========================================================================
    # PLOT PRINCIPALE: 2x2 Grid
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    
    # ========================================================================
    # PLOT 1: Euclidean Distance vs Amplitude Difference (Absolute)
    # ========================================================================
    ax = axes[0, 0]
    
    x = data['euclidean_distance']
    y = data['amplitude_diff_abs']
    
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
    
    correlations['euclidean_vs_amp_abs'] = {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'slope': coeffs[0]
    }
    
    ax.set_xlabel('Euclidean Distance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Amplitude Difference (Absolute)', fontsize=14, fontweight='bold')
    ax.set_title('Euclidean Distance vs Amplitude Difference\n(All 1024 pairs)', 
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
    # PLOT 2: Euclidean Distance vs Amplitude Difference (Relative %)
    # ========================================================================
    ax = axes[0, 1]
    
    x = data['euclidean_distance']
    y = data['amplitude_diff_rel']
    
    scatter = ax.scatter(x, y, c=base_codes, cmap='plasma', 
                        s=30, alpha=0.6, edgecolors='none')
    
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, poly(x_line), 'r--', linewidth=3, alpha=0.8,
           label=f'Linear Fit (slope={coeffs[0]:.3f})')
    
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    
    correlations['euclidean_vs_amp_rel'] = {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'slope': coeffs[0]
    }
    
    ax.set_xlabel('Euclidean Distance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Amplitude Difference (%)', fontsize=14, fontweight='bold')
    ax.set_title('Euclidean Distance vs Relative Amplitude Difference\n(All 1024 pairs)', 
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
    # PLOT 3: Cosine Distance vs Amplitude Difference (Absolute)
    # ========================================================================
    ax = axes[1, 0]
    
    x = data['cosine_distance']
    y = data['amplitude_diff_abs']
    
    scatter = ax.scatter(x, y, c=base_codes, cmap='coolwarm', 
                        s=30, alpha=0.6, edgecolors='none')
    
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, poly(x_line), 'r--', linewidth=3, alpha=0.8,
           label=f'Linear Fit (slope={coeffs[0]:.3f})')
    
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    
    correlations['cosine_vs_amp_abs'] = {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'slope': coeffs[0]
    }
    
    ax.set_xlabel('Cosine Distance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Amplitude Difference (Absolute)', fontsize=14, fontweight='bold')
    ax.set_title('Cosine Distance vs Amplitude Difference\n(All 1024 pairs)', 
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
    # PLOT 4: Cosine Distance vs Amplitude Difference (Relative %)
    # ========================================================================
    ax = axes[1, 1]
    
    x = data['cosine_distance']
    y = data['amplitude_diff_rel']
    
    scatter = ax.scatter(x, y, c=base_codes, cmap='RdYlGn', 
                        s=30, alpha=0.6, edgecolors='none')
    
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, poly(x_line), 'r--', linewidth=3, alpha=0.8,
           label=f'Linear Fit (slope={coeffs[0]:.3f})')
    
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    
    correlations['cosine_vs_amp_rel'] = {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'slope': coeffs[0]
    }
    
    ax.set_xlabel('Cosine Distance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Amplitude Difference (%)', fontsize=14, fontweight='bold')
    ax.set_title('Cosine Distance vs Relative Amplitude Difference\n(All 1024 pairs)', 
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
    plt.suptitle('Distance vs Amplitude Difference: All Code Pairs (1024 points)', 
                fontsize=17, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # ========================================================================
    # Salvataggio
    # ========================================================================
    output_path = "amplitude_difference_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Grafico salvato: {output_path}")
    
    # Log su W&B
    if use_wandb:
        wandb.log({
            "amplitude_difference_scatter": wandb.Image(fig)
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
    
    print(f"\n🔵 AMPLITUDE DIFFERENCE (ABSOLUTE):")
    print(f"   Mean: {data['amplitude_diff_abs'].mean():.4f}")
    print(f"   Std:  {data['amplitude_diff_abs'].std():.4f}")
    print(f"   Min:  {data['amplitude_diff_abs'].min():.4f}")
    print(f"   Max:  {data['amplitude_diff_abs'].max():.4f}")
    print(f"   Median: {np.median(data['amplitude_diff_abs']):.4f}")
    
    print(f"\n🟣 AMPLITUDE DIFFERENCE (RELATIVE %):")
    print(f"   Mean: {data['amplitude_diff_rel'].mean():.2f}%")
    print(f"   Std:  {data['amplitude_diff_rel'].std():.2f}%")
    print(f"   Min:  {data['amplitude_diff_rel'].min():.2f}%")
    print(f"   Max:  {data['amplitude_diff_rel'].max():.2f}%")
    print(f"   Median: {np.median(data['amplitude_diff_rel']):.2f}%")
    
    print(f"\n🟢 COSINE DISTANCE:")
    print(f"   Mean: {data['cosine_distance'].mean():.4f}")
    print(f"   Std:  {data['cosine_distance'].std():.4f}")
    
    print(f"\n🟡 EUCLIDEAN DISTANCE:")
    print(f"   Mean: {data['euclidean_distance'].mean():.4f}")
    print(f"   Std:  {data['euclidean_distance'].std():.4f}")
    
    print(f"\n{'='*70}")
    print("📈 RELAZIONI DISTANZA-AMPIEZZA")
    print(f"{'='*70}")
    
    for key, vals in correlations.items():
        metric_name = key.replace('_', ' ').title()
        print(f"\n{metric_name}:")
        print(f"   Pearson r:  {vals['pearson_r']:.4f} (p={vals['pearson_p']:.2e})")
        print(f"   Spearman ρ: {vals['spearman_r']:.4f} (p={vals['spearman_p']:.2e})")
        print(f"   Slope:      {vals['slope']:.4f}")
        
        # Interpretazione
        if abs(vals['pearson_r']) > 0.7:
            strength = "FORTE"
        elif abs(vals['pearson_r']) > 0.4:
            strength = "MODERATA"
        elif abs(vals['pearson_r']) > 0.2:
            strength = "DEBOLE"
        else:
            strength = "MOLTO DEBOLE/NESSUNA"
        
        direction = "POSITIVA" if vals['pearson_r'] > 0 else "NEGATIVA"
        print(f"   → Correlazione {strength} {direction}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("📊 SCATTER PLOT: DISTANZA vs DIFFERENZA AMPIEZZA")
    print("="*70)
    print(f"\n📂 Directory ampiezze: {AMPLITUDE_PAIRS_DIR}/")
    
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
                    "analysis_type": "amplitude_difference_scatter",
                    "amplitude_pairs_dir": AMPLITUDE_PAIRS_DIR,
                    "total_pairs": 1024,
                }
            )
            print(f"✅ W&B inizializzato")
        
        # 1. Carica dati di ampiezza
        print(f"\n{'='*70}")
        print("📥 CARICAMENTO DATI AMPIEZZE")
        print(f"{'='*70}")
        
        all_pairs_data = load_all_amplitude_pair_files(AMPLITUDE_PAIRS_DIR)
        
        # 2. Estrai dati di ampiezza
        print(f"\n📊 Estrazione dati ampiezze...")
        data = extract_amplitude_data(all_pairs_data)
        
        # 3. Calcola distanze dalle ampiezze
        print(f"\n📊 Calcolo distanze dalle ampiezze...")
        data = compute_distances_from_amplitudes(data)
        
        # 4. Crea scatter plot
        print(f"\n{'='*70}")
        print("📈 CREAZIONE SCATTER PLOTS")
        print(f"{'='*70}")
        
        correlations = plot_amplitude_difference_scatter(data, use_wandb=USE_WANDB)
        
        # 5. Statistiche
        print_detailed_statistics(data, correlations)
        
        # 6. Log metriche descrittive
        if USE_WANDB:
            wandb.log({
                'descriptive/amp_diff_abs_mean': data['amplitude_diff_abs'].mean(),
                'descriptive/amp_diff_abs_std': data['amplitude_diff_abs'].std(),
                'descriptive/amp_diff_rel_mean': data['amplitude_diff_rel'].mean(),
                'descriptive/amp_diff_rel_std': data['amplitude_diff_rel'].std(),
                'descriptive/cosine_dist_mean': data['cosine_distance'].mean(),
                'descriptive/euclidean_dist_mean': data['euclidean_distance'].mean(),
            })
        
        print(f"\n{'='*70}")
        print("✅ ANALISI COMPLETATA!")
        print(f"{'='*70}")
        print(f"\n💡 INTERPRETAZIONE:")
        print(f"   Se l'ipotesi è corretta, dovremmo vedere:")
        print(f"   - Correlazione POSITIVA tra distanza e differenza ampiezza")
        print(f"   - Codici vicini (bassa distanza) → bassa differenza ampiezza")
        print(f"   - Codici lontani (alta distanza) → alta differenza ampiezza")
        
        if USE_WANDB:
            print(f"\n🌐 W&B Dashboard: {wandb.run.url}")
            wandb.finish()
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        if USE_WANDB:
            wandb.finish()


if __name__ == "__main__":
    main()