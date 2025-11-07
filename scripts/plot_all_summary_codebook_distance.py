#!/usr/bin/env python3
"""
Visualizzazione Summary di Tutti i Codici Base

Legge tutti i file summary_codice_*.json dalla cartella "sommario_coppie"
e crea un unico grafico completo con tutte le metriche.

Uso:
    python plot_all_summaries.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

SUMMARY_DIR = "correlazione_distanze_temporali_spettrali/sommario coppie"
OUTPUT_DIR = "correlazione_distanze_temporali_spettrali/visualizations_summary"

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
        # Formato possibili:
        # - summary_codice_0.json
        # - summary_codice0.json
        filename = file_path.stem  # summary_codice_0 o summary_codice0
        
        # Prova a estrarre il numero
        try:
            # Rimuovi "summary_codice_" o "summary_codice"
            if '_' in filename:
                parts = filename.split('_')
                code_idx = int(parts[-1])  # ultimo elemento dopo split
            else:
                # summary_codice0 -> prendi i numeri alla fine
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
        'temporal_corr_min': [],
        'temporal_corr_max': [],
        'spectral_corr_mean': [],
        'spectral_corr_std': [],
        'spectral_corr_min': [],
        'spectral_corr_max': [],
        'cosine_dist_mean': [],
        'cosine_dist_std': [],
        'cosine_dist_min': [],
        'cosine_dist_max': [],
        'euclidean_dist_mean': [],
        'euclidean_dist_std': [],
        'euclidean_dist_min': [],
        'euclidean_dist_max': [],
    }
    
    for code_idx in code_indices:
        summary = summaries[code_idx]
        stats = summary['statistics']
        
        # Temporal correlation
        data['temporal_corr_mean'].append(stats['temporal_correlation']['mean'])
        data['temporal_corr_std'].append(stats['temporal_correlation']['std'])
        data['temporal_corr_min'].append(stats['temporal_correlation']['min'])
        data['temporal_corr_max'].append(stats['temporal_correlation']['max'])
        
        # Spectral correlation
        data['spectral_corr_mean'].append(stats['spectral_correlation']['mean'])
        data['spectral_corr_std'].append(stats['spectral_correlation']['std'])
        data['spectral_corr_min'].append(stats['spectral_correlation']['min'])
        data['spectral_corr_max'].append(stats['spectral_correlation']['max'])
        
        # Cosine distance
        data['cosine_dist_mean'].append(stats['cosine_distance']['mean'])
        data['cosine_dist_std'].append(stats['cosine_distance']['std'])
        data['cosine_dist_min'].append(stats['cosine_distance']['min'])
        data['cosine_dist_max'].append(stats['cosine_distance']['max'])
        
        # Euclidean distance
        data['euclidean_dist_mean'].append(stats['euclidean_distance']['mean'])
        data['euclidean_dist_std'].append(stats['euclidean_distance']['std'])
        data['euclidean_dist_min'].append(stats['euclidean_distance']['min'])
        data['euclidean_dist_max'].append(stats['euclidean_distance']['max'])
    
    # Converti liste in array
    for key in data:
        if key != 'code_indices':
            data[key] = np.array(data[key])
    
    return data

# ============================================================================
# FUNZIONI - VISUALIZZAZIONE
# ============================================================================

def plot_all_summaries(data, output_dir):
    """
    Crea un unico grafico completo con tutte le metriche per tutti i codici
    
    Layout: 2x2 grid
    - Top-Left: Temporal Correlation (mean ± std)
    - Top-Right: Spectral Correlation (mean ± std)
    - Bottom-Left: Cosine Distance (mean ± std)
    - Bottom-Right: Euclidean Distance (mean ± std)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    codes = data['code_indices']
    
    # ========================================================================
    # PLOT PRINCIPALE: 2x2 Grid
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # ========================================================================
    # PLOT 1: Temporal Correlation
    # ========================================================================
    ax = axes[0, 0]
    
    mean = data['temporal_corr_mean']
    std = data['temporal_corr_std']
    min_val = data['temporal_corr_min']
    max_val = data['temporal_corr_max']
    
    # Line plot con error bars
    ax.plot(codes, mean, 'o-', color='#2E86AB', linewidth=2, markersize=8, 
            label='Mean', zorder=3)
    ax.fill_between(codes, mean - std, mean + std, 
                     alpha=0.3, color='#2E86AB', label='Mean ± Std', zorder=2)
    
    # Min/Max range (più sottile)
    ax.fill_between(codes, min_val, max_val, 
                     alpha=0.15, color='#2E86AB', label='Min-Max Range', zorder=1)
    
    ax.set_xlabel('Base Code Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temporal Correlation', fontsize=13, fontweight='bold')
    ax.set_title('Temporal Correlation Statistics per Base Code', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlim(codes[0] - 0.5, codes[-1] + 0.5)
    
    # Stats box
    overall_mean = np.mean(mean)
    overall_std = np.std(mean)
    stats_text = f'Overall Mean: {overall_mean:.3f}\n'
    stats_text += f'Overall Std: {overall_std:.3f}\n'
    stats_text += f'Min: {np.min(min_val):.3f}\n'
    stats_text += f'Max: {np.max(max_val):.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # ========================================================================
    # PLOT 2: Spectral Correlation
    # ========================================================================
    ax = axes[0, 1]
    
    mean = data['spectral_corr_mean']
    std = data['spectral_corr_std']
    min_val = data['spectral_corr_min']
    max_val = data['spectral_corr_max']
    
    ax.plot(codes, mean, 'o-', color='#A23B72', linewidth=2, markersize=8, 
            label='Mean', zorder=3)
    ax.fill_between(codes, mean - std, mean + std, 
                     alpha=0.3, color='#A23B72', label='Mean ± Std', zorder=2)
    ax.fill_between(codes, min_val, max_val, 
                     alpha=0.15, color='#A23B72', label='Min-Max Range', zorder=1)
    
    ax.set_xlabel('Base Code Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Spectral Correlation', fontsize=13, fontweight='bold')
    ax.set_title('Spectral Correlation Statistics per Base Code', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlim(codes[0] - 0.5, codes[-1] + 0.5)
    
    # Stats box
    overall_mean = np.mean(mean)
    overall_std = np.std(mean)
    stats_text = f'Overall Mean: {overall_mean:.3f}\n'
    stats_text += f'Overall Std: {overall_std:.3f}\n'
    stats_text += f'Min: {np.min(min_val):.3f}\n'
    stats_text += f'Max: {np.max(max_val):.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # ========================================================================
    # PLOT 3: Cosine Distance
    # ========================================================================
    ax = axes[1, 0]
    
    mean = data['cosine_dist_mean']
    std = data['cosine_dist_std']
    min_val = data['cosine_dist_min']
    max_val = data['cosine_dist_max']
    
    ax.plot(codes, mean, 'o-', color='#F18F01', linewidth=2, markersize=8, 
            label='Mean', zorder=3)
    ax.fill_between(codes, mean - std, mean + std, 
                     alpha=0.3, color='#F18F01', label='Mean ± Std', zorder=2)
    ax.fill_between(codes, min_val, max_val, 
                     alpha=0.15, color='#F18F01', label='Min-Max Range', zorder=1)
    
    ax.set_xlabel('Base Code Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cosine Distance', fontsize=13, fontweight='bold')
    ax.set_title('Cosine Distance Statistics per Base Code', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(codes[0] - 0.5, codes[-1] + 0.5)
    
    # Stats box
    overall_mean = np.mean(mean)
    overall_std = np.std(mean)
    stats_text = f'Overall Mean: {overall_mean:.3f}\n'
    stats_text += f'Overall Std: {overall_std:.3f}\n'
    stats_text += f'Min: {np.min(min_val):.3f}\n'
    stats_text += f'Max: {np.max(max_val):.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # ========================================================================
    # PLOT 4: Euclidean Distance
    # ========================================================================
    ax = axes[1, 1]
    
    mean = data['euclidean_dist_mean']
    std = data['euclidean_dist_std']
    min_val = data['euclidean_dist_min']
    max_val = data['euclidean_dist_max']
    
    ax.plot(codes, mean, 'o-', color='#06A77D', linewidth=2, markersize=8, 
            label='Mean', zorder=3)
    ax.fill_between(codes, mean - std, mean + std, 
                     alpha=0.3, color='#06A77D', label='Mean ± Std', zorder=2)
    ax.fill_between(codes, min_val, max_val, 
                     alpha=0.15, color='#06A77D', label='Min-Max Range', zorder=1)
    
    ax.set_xlabel('Base Code Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Euclidean Distance', fontsize=13, fontweight='bold')
    ax.set_title('Euclidean Distance Statistics per Base Code', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(codes[0] - 0.5, codes[-1] + 0.5)
    
    # Stats box
    overall_mean = np.mean(mean)
    overall_std = np.std(mean)
    stats_text = f'Overall Mean: {overall_mean:.3f}\n'
    stats_text += f'Overall Std: {overall_std:.3f}\n'
    stats_text += f'Min: {np.min(min_val):.3f}\n'
    stats_text += f'Max: {np.max(max_val):.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # ========================================================================
    # Finalizzazione
    # ========================================================================
    plt.suptitle('Summary Statistics Across All Base Codes', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'all_summaries_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot salvato: {output_path}")
    plt.close()
    
    # ========================================================================
    # PLOT AGGIUNTIVO: Heatmap delle medie
    # ========================================================================
    plot_heatmap_comparison(data, output_dir)


def plot_heatmap_comparison(data, output_dir):
    """
    Crea una heatmap che mostra tutte le metriche medie per ogni codice
    """
    codes = data['code_indices']
    
    # Crea matrice: righe = metriche, colonne = codici
    metrics_matrix = np.array([
        data['temporal_corr_mean'],
        data['spectral_corr_mean'],
        data['cosine_dist_mean'],
        data['euclidean_dist_mean'],
    ])
    
    metric_names = [
        'Temporal Correlation',
        'Spectral Correlation',
        'Cosine Distance',
        'Euclidean Distance'
    ]
    
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Normalizza ogni riga per visualizzazione migliore
    im = ax.imshow(metrics_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    
    # Labels
    ax.set_xticks(np.arange(len(codes)))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_xticklabels(codes)
    ax.set_yticklabels(metric_names)
    
    ax.set_xlabel('Base Code Index', fontsize=13, fontweight='bold')
    ax.set_title('Mean Metrics Heatmap Across All Base Codes', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Metric Value', fontsize=11)
    
    # Aggiungi valori numerici nelle celle
    for i in range(len(metric_names)):
        for j in range(len(codes)):
            text = ax.text(j, i, f'{metrics_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'all_summaries_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap salvata: {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("📊 VISUALIZZAZIONE SUMMARY DI TUTTI I CODICI BASE")
    print("="*70)
    print(f"\n📂 Directory input: {SUMMARY_DIR}/")
    print(f"📂 Directory output: {OUTPUT_DIR}/")
    
    try:
        # 1. Carica tutti i summary
        print(f"\n{'='*70}")
        print("📥 CARICAMENTO DATI")
        print(f"{'='*70}")
        
        summaries = load_all_summaries(SUMMARY_DIR)
        
        # 2. Estrai metriche in array
        print(f"\n📊 Estrazione metriche...")
        data = extract_metrics_arrays(summaries)
        
        # 3. Crea visualizzazioni
        print(f"\n{'='*70}")
        print("📈 CREAZIONE GRAFICI")
        print(f"{'='*70}")
        
        plot_all_summaries(data, OUTPUT_DIR)
        
        # 4. Stampa statistiche globali
        print(f"\n{'='*70}")
        print("📊 STATISTICHE GLOBALI (su tutti i codici)")
        print(f"{'='*70}")
        
        print(f"\n🔵 Temporal Correlation:")
        print(f"   Overall mean: {np.mean(data['temporal_corr_mean']):.4f}")
        print(f"   Overall std: {np.std(data['temporal_corr_mean']):.4f}")
        print(f"   Global min: {np.min(data['temporal_corr_min']):.4f}")
        print(f"   Global max: {np.max(data['temporal_corr_max']):.4f}")
        
        print(f"\n🟣 Spectral Correlation:")
        print(f"   Overall mean: {np.mean(data['spectral_corr_mean']):.4f}")
        print(f"   Overall std: {np.std(data['spectral_corr_mean']):.4f}")
        print(f"   Global min: {np.min(data['spectral_corr_min']):.4f}")
        print(f"   Global max: {np.max(data['spectral_corr_max']):.4f}")
        
        print(f"\n🟠 Cosine Distance:")
        print(f"   Overall mean: {np.mean(data['cosine_dist_mean']):.4f}")
        print(f"   Overall std: {np.std(data['cosine_dist_mean']):.4f}")
        print(f"   Global min: {np.min(data['cosine_dist_min']):.4f}")
        print(f"   Global max: {np.max(data['cosine_dist_max']):.4f}")
        
        print(f"\n🟢 Euclidean Distance:")
        print(f"   Overall mean: {np.mean(data['euclidean_dist_mean']):.4f}")
        print(f"   Overall std: {np.std(data['euclidean_dist_mean']):.4f}")
        print(f"   Global min: {np.min(data['euclidean_dist_min']):.4f}")
        print(f"   Global max: {np.max(data['euclidean_dist_max']):.4f}")
        
        print(f"\n{'='*70}")
        print("✅ VISUALIZZAZIONE COMPLETATA!")
        print(f"{'='*70}")
        print(f"\n📁 Grafici salvati in: {OUTPUT_DIR}/")
        print(f"   - all_summaries_comparison.png (4 plot)")
        print(f"   - all_summaries_heatmap.png")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()