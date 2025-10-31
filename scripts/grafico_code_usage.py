#!/usr/bin/env python3
"""
Grafico Usage Percentage vs Embeddings Size

Crea UN SINGOLO grafico che mostra la percentuale di utilizzo del codebook
in funzione della dimensione (numero di embeddings).

Uso:
    python scripts/plot_usage_vs_embeddings.py
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
    parts = filename.replace('.json', '').split('_')
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def plot_usage_percentage(all_data, output_dir):
    """
    Crea grafico: Usage Percentage vs Embeddings Size
    """
    # Estrai dati
    embeddings_sizes = []
    usage_percentages = []
    
    for filename, data in all_data.items():
        train_results = data['results']['train']
        
        num_embeddings = train_results['num_embeddings']
        usage_pct = train_results['usage_percentage']
        
        embeddings_sizes.append(num_embeddings)
        usage_percentages.append(usage_pct)
    
    # Ordina per embeddings size
    sorted_indices = np.argsort(embeddings_sizes)
    embeddings_sizes = [embeddings_sizes[i] for i in sorted_indices]
    usage_percentages = [usage_percentages[i] for i in sorted_indices]
    
    # Crea figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot con linea e marker
    ax.plot(embeddings_sizes, usage_percentages, 
            'o-', linewidth=3, markersize=12, 
            color='steelblue', markerfacecolor='darkblue',
            markeredgecolor='black', markeredgewidth=1.5)
    
    # Aggiungi valori sopra i punti
    for x, y in zip(embeddings_sizes, usage_percentages):
        ax.annotate(f'{y:.1f}%', 
                   (x, y), 
                   textcoords="offset points",
                   xytext=(0, 12), 
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
    ax.set_title('Codebook Usage Percentage vs Size', 
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
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Box con statistiche
    stats_text = f'Configurations: {len(embeddings_sizes)}\n'
    stats_text += f'Max usage: {max(usage_percentages):.1f}% ({embeddings_sizes[usage_percentages.index(max(usage_percentages))]} emb)\n'
    stats_text += f'Min usage: {min(usage_percentages):.1f}% ({embeddings_sizes[usage_percentages.index(min(usage_percentages))]} emb)\n'
    stats_text += f'Mean usage: {np.mean(usage_percentages):.1f}%'
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Salva
    output_path = output_dir / 'usage_percentage_vs_embeddings.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafico salvato: {output_path}")
    
    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("📊 GRAFICO: USAGE PERCENTAGE vs EMBEDDINGS SIZE")
    print("="*70)
    print(f"\n📂 Input directory: {CODE_USAGE_DIR}")
    print(f"📂 Output directory: {OUTPUT_DIR}")
    
    # Trova tutti i file JSON
    json_files = sorted(CODE_USAGE_DIR.glob("*.json"))
    
    if not json_files:
        print(f"\n❌ Nessun file JSON trovato in {CODE_USAGE_DIR}")
        print(f"   Assicurati di aver eseguito codebook_usage.py prima!")
        return
    
    print(f"\n🔍 File JSON trovati: {len(json_files)}")
    
    # Carica tutti i dati
    all_data = {}
    
    for json_path in json_files:
        filename = json_path.name
        embeddings_size = extract_embeddings_from_filename(filename)
        
        if embeddings_size is None:
            print(f"⚠️  Impossibile estrarre embeddings size da: {filename}")
            continue
        
        data = load_code_usage_data(json_path)
        all_data[filename] = data
        
        train_results = data['results']['train']
        usage_pct = train_results['usage_percentage']
        
        print(f"   ✅ {embeddings_size:4d} embeddings: {usage_pct:6.2f}% usage")
    
    if not all_data:
        print(f"\n❌ Nessun dato valido caricato")
        return
    
    print(f"\n📊 Creazione grafico...")
    
    # Crea IL grafico
    output_path = plot_usage_percentage(all_data, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("✅ GRAFICO COMPLETATO!")
    print(f"📂 Salvato in: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()