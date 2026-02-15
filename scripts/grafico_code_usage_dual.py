#!/usr/bin/env python3
"""
Grafico Usage Percentage vs Embeddings Size - DUAL CODEBOOK

Crea un grafico che mostra la percentuale di utilizzo dei DUE codebook
(Active e Inactive) in funzione della dimensione (numero di embeddings).

Uso:
    python scripts/grafico_code_usage_dual.py
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


def is_dual_model(data):
    """Verifica se il modello è dual codebook"""
    metadata = data.get('metadata', {})
    return metadata.get('model_type') == 'DualCalciumVQVAE'


def plot_dual_usage_percentage(all_data, output_dir):
    """
    Crea grafico: Usage Percentage vs Embeddings Size per DUAL CODEBOOK
    Mostra Active, Inactive e Combined
    """
    # Estrai dati
    embeddings_sizes = []
    usage_active = []
    usage_inactive = []
    usage_combined = []
    
    for filename, data in all_data.items():
        if not is_dual_model(data):
            continue
            
        train_results = data['results']['train']
        
        num_embeddings_total = train_results['num_embeddings_total']
        usage_active_pct = train_results['active']['usage_percentage']
        usage_inactive_pct = train_results['inactive']['usage_percentage']
        usage_combined_pct = train_results['combined']['usage_percentage']
        
        embeddings_sizes.append(num_embeddings_total)
        usage_active.append(usage_active_pct)
        usage_inactive.append(usage_inactive_pct)
        usage_combined.append(usage_combined_pct)
    
    if not embeddings_sizes:
        print("❌ Nessun dato dual codebook trovato!")
        return None
    
    # Ordina per embeddings size
    sorted_indices = np.argsort(embeddings_sizes)
    embeddings_sizes = [embeddings_sizes[i] for i in sorted_indices]
    usage_active = [usage_active[i] for i in sorted_indices]
    usage_inactive = [usage_inactive[i] for i in sorted_indices]
    usage_combined = [usage_combined[i] for i in sorted_indices]
    
    # Crea figura
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Plot Active codebook
    ax.plot(embeddings_sizes, usage_active, 
            'o-', linewidth=3, markersize=12, 
            color='#2ecc71', markerfacecolor='#27ae60',
            markeredgecolor='black', markeredgewidth=1.5,
            label='Active Codebook', zorder=3)
    
    # Plot Inactive codebook
    ax.plot(embeddings_sizes, usage_inactive, 
            's-', linewidth=3, markersize=12, 
            color='#e74c3c', markerfacecolor='#c0392b',
            markeredgecolor='black', markeredgewidth=1.5,
            label='Inactive Codebook', zorder=3)
    
    # Plot Combined
    ax.plot(embeddings_sizes, usage_combined, 
            '^-', linewidth=3, markersize=12, 
            color='#3498db', markerfacecolor='#2980b9',
            markeredgecolor='black', markeredgewidth=1.5,
            label='Combined (Total)', zorder=3, linestyle='--')
    
    # Aggiungi valori sopra i punti (solo per combined per non sovraffollare)
    for x, y in zip(embeddings_sizes, usage_combined):
        ax.annotate(f'{y:.1f}%', 
                   (x, y), 
                   textcoords="offset points",
                   xytext=(0, 15), 
                   ha='center',
                   fontsize=10,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='lightblue', 
                           alpha=0.8,
                           edgecolor='black'))
    
    # Linee di riferimento
    ax.axhline(100, color='green', linestyle=':', linewidth=2, 
              alpha=0.4, label='100% Usage')
    ax.axhline(50, color='orange', linestyle=':', linewidth=2, 
              alpha=0.4, label='50% Usage')
    
    # Labels e titolo
    ax.set_xlabel('Total Codebook Size (Number of Embeddings)', 
                 fontsize=15, fontweight='bold')
    ax.set_ylabel('Usage Percentage (%)', fontsize=15, fontweight='bold')
    ax.set_title('Dual Codebook Usage Percentage vs Size\n(Active vs Inactive Codebooks)', 
                fontsize=17, fontweight='bold', pad=20)
    
    # Griglia
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, zorder=0)
    
    # Scala logaritmica per x
    ax.set_xscale('log')
    
    # Limiti y
    max_usage = max(max(usage_active), max(usage_inactive), max(usage_combined))
    ax.set_ylim([0, min(105, max_usage * 1.1)])
    
    # Ticks personalizzati per x
    ax.set_xticks(embeddings_sizes)
    ax.set_xticklabels([str(s) for s in embeddings_sizes], fontsize=11)
    
    # Legend
    ax.legend(loc='best', fontsize=12, framealpha=0.95, 
             fancybox=True, shadow=True)
    
    # Box con statistiche
    stats_text = f'Configurations: {len(embeddings_sizes)}\n'
    stats_text += f'\nActive Codebook:\n'
    stats_text += f'  Max: {max(usage_active):.1f}% ({embeddings_sizes[usage_active.index(max(usage_active))]} emb)\n'
    stats_text += f'  Min: {min(usage_active):.1f}% ({embeddings_sizes[usage_active.index(min(usage_active))]} emb)\n'
    stats_text += f'  Mean: {np.mean(usage_active):.1f}%\n'
    stats_text += f'\nInactive Codebook:\n'
    stats_text += f'  Max: {max(usage_inactive):.1f}% ({embeddings_sizes[usage_inactive.index(max(usage_inactive))]} emb)\n'
    stats_text += f'  Min: {min(usage_inactive):.1f}% ({embeddings_sizes[usage_inactive.index(min(usage_inactive))]} emb)\n'
    stats_text += f'  Mean: {np.mean(usage_inactive):.1f}%\n'
    stats_text += f'\nCombined:\n'
    stats_text += f'  Mean: {np.mean(usage_combined):.1f}%'
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           fontsize=10,
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, 
                    edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    
    # Salva
    output_path = output_dir / 'usage_percentage_vs_embeddings_dual.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafico salvato: {output_path}")
    
    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("📊 GRAFICO: DUAL CODEBOOK USAGE PERCENTAGE vs EMBEDDINGS SIZE")
    print("="*70)
    print(f"\n📂 Input directory: {CODE_USAGE_DIR}")
    print(f"📂 Output directory: {OUTPUT_DIR}")
    
    # Trova tutti i file JSON
    json_files = sorted(CODE_USAGE_DIR.glob("*.json"))
    
    if not json_files:
        print(f"\n❌ Nessun file JSON trovato in {CODE_USAGE_DIR}")
        print(f"   Assicurati di aver eseguito codebook_usage_dual.py prima!")
        return
    
    print(f"\n🔍 File JSON trovati: {len(json_files)}")
    
    # Carica tutti i dati e filtra solo dual
    all_data = {}
    dual_count = 0
    
    for json_path in json_files:
        filename = json_path.name
        
        try:
            data = load_code_usage_data(json_path)
            
            # Filtra solo dual codebook
            if is_dual_model(data):
                all_data[filename] = data
                dual_count += 1
                
                train_results = data['results']['train']
                num_embeddings_total = train_results['num_embeddings_total']
                usage_active = train_results['active']['usage_percentage']
                usage_inactive = train_results['inactive']['usage_percentage']
                usage_combined = train_results['combined']['usage_percentage']
                
                print(f"   ✅ {num_embeddings_total:4d} embeddings (dual): "
                      f"Active={usage_active:5.2f}%, "
                      f"Inactive={usage_inactive:5.2f}%, "
                      f"Combined={usage_combined:5.2f}%")
            else:
                print(f"   ⏭️  {filename}: Single codebook (skip)")
        except Exception as e:
            print(f"   ⚠️  Errore caricando {filename}: {e}")
            continue
    
    if not all_data:
        print(f"\n❌ Nessun modello dual codebook trovato!")
        print(f"   Assicurati di aver eseguito codebook_usage_dual.py per generare i JSON dual!")
        return
    
    print(f"\n📊 Modelli dual trovati: {dual_count}")
    print(f"\n📊 Creazione grafico...")
    
    # Crea il grafico
    output_path = plot_dual_usage_percentage(all_data, OUTPUT_DIR)
    
    if output_path:
        print("\n" + "="*70)
        print("✅ GRAFICO COMPLETATO!")
        print(f"📂 Salvato in: {output_path}")
        print("="*70)
    else:
        print("\n❌ Errore nella creazione del grafico")


if __name__ == "__main__":
    main()



