#!/usr/bin/env python3
"""
Grafico Usage Percentage vs Embeddings Size - DUAL CODEBOOK CONSTRAINED

Crea UN SINGOLO grafico che mostra la percentuale di utilizzo del codebook
in funzione della dimensione (numero di embeddings) per i modelli DUAL CONSTRAINED.

Uso:
    python scripts/grafico_code_usage_dual_constrained.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CODE_USAGE_DIR = Path("code_usage")
OUTPUT_DIR = Path("code_usage_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pattern per filtrare solo i modelli dual constrained
FILTER_PATTERN = "best_dual_model_*_constrained.json"

# ============================================================================
# FUNZIONI
# ============================================================================

def load_code_usage_data(json_path):
    """Carica dati da JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_embeddings_from_filename(filename):
    """Estrae il numero di embeddings dal nome del file"""
    # Esempio: best_dual_model_32_constrained.json -> 32
    parts = filename.replace('.json', '').split('_')
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def plot_usage_percentage_dual(all_data, output_dir):
    """
    Crea grafico: Usage Percentage vs Embeddings Size per modelli DUAL CONSTRAINED
    Mostra: Combined, Active, Inactive
    """
    # Estrai dati
    embeddings_sizes = []
    usage_percentages_combined = []
    usage_percentages_active = []
    usage_percentages_inactive = []
    
    for filename, data in all_data.items():
        train_results = data['results']['train']
        
        num_embeddings = train_results['num_embeddings_total']
        
        # Combined usage
        combined_usage = train_results['combined']['usage_percentage']
        
        # Active usage
        active_usage = train_results['active']['usage_percentage']
        
        # Inactive usage
        inactive_usage = train_results['inactive']['usage_percentage']
        
        embeddings_sizes.append(num_embeddings)
        usage_percentages_combined.append(combined_usage)
        usage_percentages_active.append(active_usage)
        usage_percentages_inactive.append(inactive_usage)
    
    # Ordina per embeddings size
    sorted_indices = np.argsort(embeddings_sizes)
    embeddings_sizes = [embeddings_sizes[i] for i in sorted_indices]
    usage_percentages_combined = [usage_percentages_combined[i] for i in sorted_indices]
    usage_percentages_active = [usage_percentages_active[i] for i in sorted_indices]
    usage_percentages_inactive = [usage_percentages_inactive[i] for i in sorted_indices]
    
    # Crea figura
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Plot Combined (linea principale)
    ax.plot(embeddings_sizes, usage_percentages_combined, 
            'o-', linewidth=3, markersize=12, 
            color='steelblue', markerfacecolor='darkblue',
            markeredgecolor='black', markeredgewidth=1.5,
            label='Combined (Total)')
    
    # Plot Active
    ax.plot(embeddings_sizes, usage_percentages_active, 
            's-', linewidth=2, markersize=10, 
            color='green', markerfacecolor='lightgreen',
            markeredgecolor='black', markeredgewidth=1,
            label='Active Codebook', alpha=0.8)
    
    # Plot Inactive
    ax.plot(embeddings_sizes, usage_percentages_inactive, 
            '^-', linewidth=2, markersize=10, 
            color='red', markerfacecolor='lightcoral',
            markeredgecolor='black', markeredgewidth=1,
            label='Inactive Codebook', alpha=0.8)
    
    # Aggiungi valori sopra i punti (solo per combined)
    for x, y in zip(embeddings_sizes, usage_percentages_combined):
        ax.annotate(f'{y:.1f}%', 
                   (x, y), 
                   textcoords="offset points",
                   xytext=(0, 15), 
                   ha='center',
                   fontsize=11,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                           facecolor='yellow', 
                           alpha=0.7,
                           edgecolor='black'))
    
    # Linea al 100%
    ax.axhline(100, color='green', linestyle='--', linewidth=2, 
              alpha=0.5, label='100% Usage')
    
    # Linea al 50%
    ax.axhline(50, color='orange', linestyle='--', linewidth=2, 
              alpha=0.5, label='50% Usage')
    
    # Labels e titolo
    ax.set_xlabel('Codebook Size (Number of Embeddings)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Usage Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Dual Codebook Usage Percentage vs Size (Constrained Models)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Griglia
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    # Scala logaritmica per x (più leggibile)
    ax.set_xscale('log')
    
    # Limiti y
    ax.set_ylim([0, 105])
    
    # Ticks personalizzati per x
    ax.set_xticks(embeddings_sizes)
    ax.set_xticklabels([str(s) for s in embeddings_sizes], fontsize=11)
    
    # Legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9, ncol=2)
    
    # Box con statistiche
    stats_text = f'Dual Constrained Models: {len(embeddings_sizes)}\n\n'
    stats_text += f'COMBINED:\n'
    stats_text += f'  Max: {max(usage_percentages_combined):.1f}% ({embeddings_sizes[usage_percentages_combined.index(max(usage_percentages_combined))]} emb)\n'
    stats_text += f'  Min: {min(usage_percentages_combined):.1f}% ({embeddings_sizes[usage_percentages_combined.index(min(usage_percentages_combined))]} emb)\n'
    stats_text += f'  Mean: {np.mean(usage_percentages_combined):.1f}%\n\n'
    stats_text += f'ACTIVE:\n'
    stats_text += f'  Mean: {np.mean(usage_percentages_active):.1f}%\n\n'
    stats_text += f'INACTIVE:\n'
    stats_text += f'  Mean: {np.mean(usage_percentages_inactive):.1f}%'
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Salva
    output_path = output_dir / 'usage_percentage_vs_embeddings_dual_constrained.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafico salvato: {output_path}")
    
    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("📊 GRAFICO: USAGE PERCENTAGE vs EMBEDDINGS SIZE (DUAL CONSTRAINED)")
    print("="*70)
    print(f"\n📂 Input directory: {CODE_USAGE_DIR}")
    print(f"📂 Output directory: {OUTPUT_DIR}")
    print(f"🔍 Pattern: {FILTER_PATTERN}")
    
    # Trova tutti i file JSON che corrispondono al pattern
    all_json_files = sorted(CODE_USAGE_DIR.glob("*.json"))
    json_files = [f for f in all_json_files if "dual" in f.name and "constrained" in f.name]
    
    if not json_files:
        print(f"\n❌ Nessun file JSON dual constrained trovato in {CODE_USAGE_DIR}")
        print(f"   Cerca file con pattern: {FILTER_PATTERN}")
        return
    
    print(f"\n🔍 File JSON dual constrained trovati: {len(json_files)}")
    
    # Carica tutti i dati
    all_data = {}
    
    for json_path in json_files:
        filename = json_path.name
        embeddings_size = extract_embeddings_from_filename(filename)
        
        if embeddings_size is None:
            print(f"⚠️  Impossibile estrarre embeddings size da: {filename}")
            continue
        
        try:
            data = load_code_usage_data(json_path)
            
            # Verifica che sia un file dual (ha 'active' e 'inactive')
            if 'results' in data and 'train' in data['results']:
                train_results = data['results']['train']
                if 'active' in train_results and 'inactive' in train_results:
                    all_data[filename] = data
                    
                    combined_usage = train_results['combined']['usage_percentage']
                    active_usage = train_results['active']['usage_percentage']
                    inactive_usage = train_results['inactive']['usage_percentage']
                    
                    print(f"   ✅ {embeddings_size:4d} embeddings: Combined={combined_usage:6.2f}%, "
                          f"Active={active_usage:6.2f}%, Inactive={inactive_usage:6.2f}%")
                else:
                    print(f"⚠️  {filename} non è un file dual codebook valido")
            else:
                print(f"⚠️  {filename} non ha la struttura attesa")
        except Exception as e:
            print(f"❌ Errore caricando {filename}: {e}")
            continue
    
    if not all_data:
        print(f"\n❌ Nessun dato valido caricato")
        return
    
    print(f"\n📊 Creazione grafico...")
    
    # Crea IL grafico
    output_path = plot_usage_percentage_dual(all_data, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("✅ GRAFICO COMPLETATO!")
    print(f"📂 Salvato in: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()

