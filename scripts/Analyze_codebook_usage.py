#!/usr/bin/env python3
"""
Analisi Utilizzo Codebook VQ-VAE - SINGLE NEURON

Questo script analizza quali codici del codebook vengono effettivamente
utilizzati durante la ricostruzione per modelli VQ-VAE single neuron.

Output formato:
    0: 27    <- codice 0 usato 27 volte
    1: 0     <- codice 1 NON usato
    2: 125   <- codice 2 usato 125 volte
    ...
    Codici usati: 2 / 3 (66.67%)

🔧 COME USARE:
    1. Modifica CHECKPOINT_PATH in cima al file
    2. Modifica MODEL_CONFIG per corrispondere al tuo modello
    3. Lancia: python scripts/analyze_codebook_usage.py
    
Esempio:
    # In cima al file, cambia:
    CHECKPOINT_PATH = "results/best_single_neuron_model_1024_pt2.pth"
    MODEL_CONFIG = {
        'num_embeddings': 1024,  # <- Numero codici nel tuo modello
        'embedding_dim': 128,
        ...
    }
    
    # Poi lancia semplicemente:
    python scripts/analyze_codebook_usage.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import json
from datetime import datetime

from models.vqvae import CalciumVQVAE
from datasets.calcium import create_unified_dataloaders, TRAINING_SESSION_IDS

# ============================================================================
# CONFIGURAZIONE DIRETTA - MODIFICA QUI!
# ============================================================================

# 🔧 CHECKPOINT DA ANALIZZARE - MODIFICA QUESTO!
CHECKPOINT_PATH = "results/best_single_neuron_model_1024_pt2.pth"

# 🔧 CONFIGURAZIONE MODELLO - DEVE CORRISPONDERE AL CHECKPOINT!
MODEL_CONFIG = {
    'num_neurons': 1,           
    'num_hiddens': 128,         
    'num_residual_layers': 3,   
    'num_residual_hiddens': 64, 
    'num_embeddings': 1024,       # ← Numero codici del tuo modello
    'embedding_dim': 128,         # ← Deve corrispondere al checkpoint
    'commitment_cost': 0.5,       
    'dropout_rate': 0.0,        
    'use_quantizer': True,
}

# 🔧 PARAMETRI ANALISI
BATCH_SIZE = 64
NUM_SAMPLES = None  # None = analizza tutto il dataset, oppure es. 500 per test veloce
SAVE_DIR = "codebook_analysis"

# Parametri dataset
DEFAULT_WINDOW_SIZE = 60
DEFAULT_STRIDE = 50

# ============================================================================
# FUNZIONI PRINCIPALI
# ============================================================================

def load_single_neuron_model(checkpoint_path, device):
    """
    Carica modello SINGLE NEURON da checkpoint .pth
    
    Returns:
        model: Modello caricato
        num_embeddings: Numero di codici nel codebook
        checkpoint_info: Info aggiuntive
    """
    print(f"\n🔄 Caricando checkpoint: {checkpoint_path}")
    print(f"   Path: {os.path.abspath(checkpoint_path)}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint non trovato: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Estrai num_embeddings dal checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    if 'vector_quantization.embedding.weight' in state_dict:
        num_embeddings = state_dict['vector_quantization.embedding.weight'].shape[0]
        embedding_dim_actual = state_dict['vector_quantization.embedding.weight'].shape[1]
        print(f"\n📊 Codebook rilevato:")
        print(f"   Num embeddings (codici): {num_embeddings}")
        print(f"   Embedding dim: {embedding_dim_actual}")
    else:
        raise ValueError("❌ Impossibile trovare codebook nel checkpoint!")
    
    # Verifica se è davvero single neuron
    if 'encoder._conv_1.weight' in state_dict:
        in_channels = state_dict['encoder._conv_1.weight'].shape[1]
        if in_channels != 1:
            print(f"⚠️  ATTENZIONE: Questo modello ha {in_channels} neuroni, non 1!")
            response = input(f"   Vuoi continuare comunque? (y/n): ")
            if response.lower() != 'y':
                raise ValueError("Operazione annullata dall'utente")
    
    # Crea modello con configurazione single neuron
    model = CalciumVQVAE(
        num_neurons=MODEL_CONFIG['num_neurons'],
        num_hiddens=MODEL_CONFIG['num_hiddens'],
        num_residual_layers=MODEL_CONFIG['num_residual_layers'],
        num_residual_hiddens=MODEL_CONFIG['num_residual_hiddens'],
        num_embeddings=num_embeddings,  # ← Dal checkpoint
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        commitment_cost=MODEL_CONFIG['commitment_cost'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        use_quantizer=MODEL_CONFIG['use_quantizer']
    )
    
    # Carica pesi
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modello single neuron caricato")
    
    # Info aggiuntive
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'best_correlation': checkpoint.get('best_correlation', None),
    }
    
    return model, num_embeddings, checkpoint_info


def analyze_codebook_usage(model, dataloader, device, num_samples=None):
    """
    Analizza utilizzo del codebook durante forward pass
    
    Returns:
        usage_stats: Dizionario con statistiche dettagliate
    """
    print(f"\n🔍 Analizzando utilizzo codebook...")
    
    model.eval()
    
    # Contatore per codici utilizzati
    code_counter = Counter()
    
    total_samples = 0
    total_timesteps = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            neural_data = batch_data.to(device)
            batch_size = neural_data.shape[0]
            
            # Forward pass
            vq_loss, recon, perplexity, encodings, encoding_indices = model(neural_data)
            
            # ✅ ESTRAI encoding_indices
            if encoding_indices is not None:
                indices_flat = encoding_indices.flatten().cpu().numpy()
                
                # Conta occorrenze
                for idx in indices_flat:
                    code_counter[int(idx)] += 1
                
                total_timesteps += len(indices_flat)
            else:
                print(f"⚠️  Batch {batch_idx}: encoding_indices è None!")
            
            total_samples += batch_size
            
            # Print progress ogni 50 batch
            if (batch_idx + 1) % 50 == 0:
                print(f"   Processati {batch_idx + 1} batch, {total_samples} samples...")
            
            # Check limite samples
            if num_samples is not None and total_samples >= num_samples:
                print(f"   Raggiunto limite di {num_samples} samples")
                break
    
    print(f"\n✅ Analisi completata:")
    print(f"   Samples totali: {total_samples}")
    print(f"   Timesteps totali: {total_timesteps}")
    
    # Calcola statistiche
    num_codes_total = model.vector_quantization.n_e
    num_codes_used = len(code_counter)
    usage_percentage = (num_codes_used / num_codes_total) * 100
    
    print(f"   Codici unici utilizzati: {num_codes_used}/{num_codes_total} ({usage_percentage:.2f}%)")
    
    # Ordina per frequenza
    sorted_codes = sorted(code_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Top 10 e Bottom 10
    top_10 = sorted_codes[:10]
    bottom_used = sorted_codes[-10:] if len(sorted_codes) >= 10 else sorted_codes
    
    # Codici MAI usati
    unused_codes = set(range(num_codes_total)) - set(code_counter.keys())
    num_unused = len(unused_codes)
    
    # Statistiche di distribuzione
    if code_counter:
        usage_counts = list(code_counter.values())
        mean_usage = np.mean(usage_counts)
        std_usage = np.std(usage_counts)
        median_usage = np.median(usage_counts)
        max_usage = np.max(usage_counts)
        min_usage = np.min(usage_counts)
    else:
        mean_usage = std_usage = median_usage = max_usage = min_usage = 0
    
    usage_stats = {
        'num_codes_total': num_codes_total,
        'num_codes_used': num_codes_used,
        'num_codes_unused': num_unused,
        'usage_percentage': usage_percentage,
        'code_counter': dict(code_counter),
        'sorted_codes': sorted_codes,
        'top_10': top_10,
        'bottom_used': bottom_used,
        'unused_codes': sorted(list(unused_codes)),
        'total_samples': total_samples,
        'total_timesteps': total_timesteps,
        'mean_usage': mean_usage,
        'std_usage': std_usage,
        'median_usage': median_usage,
        'max_usage': max_usage,
        'min_usage': min_usage,
    }
    
    return usage_stats


def print_usage_report(usage_stats, checkpoint_path):
    """
    Stampa report dettagliato dell'utilizzo del codebook
    
    Formato richiesto:
        0: 27    <- codice 0 usato 27 volte
        1: 0     <- codice 1 NON usato
        2: 125   <- codice 2 usato 125 volte
        ...
        Codici usati: 2 / 3 (66.67%)
    """
    
    print("\n" + "="*70)
    print(f"📊 REPORT UTILIZZO CODEBOOK")
    print("="*70)
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Totale codici: {usage_stats['num_codes_total']}")
    print(f"Codici usati: {usage_stats['num_codes_used']}")
    print(f"Codici NON usati: {usage_stats['num_codes_unused']}")
    print(f"Percentuale utilizzo: {usage_stats['usage_percentage']:.2f}%")
    print("="*70)
    
    # ✅ FORMATO RICHIESTO: Tutti i codici con loro conteggio
    print(f"\n📋 UTILIZZO PER CODICE:")
    print("-"*70)
    
    code_counter = usage_stats['code_counter']
    num_total = usage_stats['num_codes_total']
    
    for code_idx in range(num_total):
        count = code_counter.get(code_idx, 0)
        
        if count > 0:
            print(f"{code_idx}: {count}")
        else:
            print(f"{code_idx}: 0")
    
    print("-"*70)
    print(f"\n✅ SUMMARY: Codici usati {usage_stats['num_codes_used']} / {num_total} ({usage_stats['usage_percentage']:.2f}%)")
    
    # Statistiche aggiuntive
    if usage_stats['num_codes_used'] > 0:
        print(f"\n📊 STATISTICHE DISTRIBUZIONE:")
        print(f"   Mean usage: {usage_stats['mean_usage']:.1f}")
        print(f"   Median usage: {usage_stats['median_usage']:.1f}")
        print(f"   Std usage: {usage_stats['std_usage']:.1f}")
        print(f"   Max usage: {usage_stats['max_usage']}")
        print(f"   Min usage (escluso 0): {usage_stats['min_usage']}")
        
        # Top 5 codici più usati
        print(f"\n🔝 TOP 5 CODICI PIÙ USATI:")
        for i, (code_idx, count) in enumerate(usage_stats['top_10'][:5], 1):
            percentage = (count / usage_stats['total_timesteps']) * 100
            print(f"   {i}. Codice {code_idx}: {count} volte ({percentage:.2f}%)")
    
    print("="*70)


def visualize_usage(usage_stats, save_path):
    """
    Crea visualizzazioni dell'utilizzo del codebook
    """
    
    num_codes = usage_stats['num_codes_total']
    code_counter = usage_stats['code_counter']
    
    # Array con conteggi (0 per codici non usati)
    usage_array = np.array([code_counter.get(i, 0) for i in range(num_codes)])
    
    # ========================================
    # FIGURA 1: Bar plot dell'utilizzo
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Codebook Usage Analysis ({num_codes} codes)', fontsize=16, fontweight='bold')
    
    # Plot 1: Bar plot di TUTTI i codici
    ax = axes[0, 0]
    colors = ['red' if count == 0 else 'skyblue' for count in usage_array]
    ax.bar(range(num_codes), usage_array, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Code Index')
    ax.set_ylabel('Usage Count')
    ax.set_title(f'Usage per Code (Red = Unused)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Histogram della distribuzione
    ax = axes[0, 1]
    used_counts = [count for count in usage_array if count > 0]
    if used_counts:
        ax.hist(used_counts, bins=min(30, len(used_counts)), color='green', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(used_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(used_counts):.1f}')
        ax.axvline(np.median(used_counts), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(used_counts):.1f}')
    ax.set_xlabel('Usage Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Usage (Excluding Unused)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative usage
    ax = axes[1, 0]
    sorted_usage = np.sort(usage_array)[::-1]  # Ordine decrescente
    cumsum = np.cumsum(sorted_usage)
    cumsum_pct = (cumsum / cumsum[-1]) * 100
    
    ax.plot(range(len(cumsum)), cumsum_pct, 'b-', linewidth=2)
    ax.axhline(y=80, color='r', linestyle='--', label='80%')
    ax.axhline(y=95, color='orange', linestyle='--', label='95%')
    ax.set_xlabel('Top N Codes')
    ax.set_ylabel('Cumulative Usage (%)')
    ax.set_title('Cumulative Usage Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Pie chart usati vs non usati
    ax = axes[1, 1]
    sizes = [usage_stats['num_codes_used'], usage_stats['num_codes_unused']]
    labels = [f"Used\n({usage_stats['num_codes_used']})", 
              f"Unused\n({usage_stats['num_codes_unused']})"]
    colors_pie = ['lightgreen', 'lightcoral']
    explode = (0.05, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title('Overall Usage')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizzazione salvata: {save_path}")


def save_json_report(usage_stats, checkpoint_path, save_path):
    """
    Salva report in formato JSON
    """
    
    report = {
        'checkpoint': os.path.basename(checkpoint_path),
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'num_codes_total': usage_stats['num_codes_total'],
            'num_codes_used': usage_stats['num_codes_used'],
            'num_codes_unused': usage_stats['num_codes_unused'],
            'usage_percentage': usage_stats['usage_percentage'],
            'total_samples': usage_stats['total_samples'],
            'total_timesteps': usage_stats['total_timesteps'],
        },
        'statistics': {
            'mean_usage': usage_stats['mean_usage'],
            'std_usage': usage_stats['std_usage'],
            'median_usage': usage_stats['median_usage'],
            'max_usage': usage_stats['max_usage'],
            'min_usage': usage_stats['min_usage'],
        },
        'usage_per_code': usage_stats['code_counter'],
        'top_10_codes': [{'code': int(c), 'count': int(cnt)} for c, cnt in usage_stats['top_10']],
        'unused_codes': usage_stats['unused_codes'],
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Report JSON salvato: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main function - usa configurazione definita in cima al file
    NON serve passare argomenti da terminale!
    """
    
    print("="*70)
    print("🔬 ANALISI CODEBOOK USAGE - SINGLE NEURON")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Num samples: {NUM_SAMPLES if NUM_SAMPLES else 'Tutti'}")
    print(f"   Save dir: {SAVE_DIR}")
    print(f"\n📊 Model config:")
    for key, value in MODEL_CONFIG.items():
        print(f"   {key}: {value}")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Crea directory output
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # 1. Carica modello
        model, num_embeddings, checkpoint_info = load_single_neuron_model(
            CHECKPOINT_PATH, device
        )
        
        print(f"\n📋 Info checkpoint:")
        print(f"   Epoca: {checkpoint_info['epoch']}")
        if checkpoint_info['best_correlation'] is not None:
            print(f"   Best correlation: {checkpoint_info['best_correlation']:.4f}")
        
        # 2. Carica dataset (SINGLE NEURON)
        print(f"\n📊 Caricando dataset single neuron...")
        train_loader, test_loader, dataset_info = create_unified_dataloaders(
            train_session_ids=TRAINING_SESSION_IDS,
            batch_size=BATCH_SIZE,
            test_split=0.2,
            window_size=DEFAULT_WINDOW_SIZE,
            stride=DEFAULT_STRIDE,
            min_neurons=1,  # ✅ SINGLE NEURON
            num_workers=0,
            use_cross_session_test=False
        )
        
        print(f"✅ Dataset caricato:")
        print(f"   Train samples: {dataset_info['train_samples']}")
        print(f"   Test samples: {dataset_info['test_samples']}")
        
        # 3. Analizza utilizzo
        usage_stats = analyze_codebook_usage(
            model, train_loader, device, 
            num_samples=NUM_SAMPLES
        )
        
        # 4. Print report
        print_usage_report(usage_stats, CHECKPOINT_PATH)
        
        # 5. Visualizza
        checkpoint_name = Path(CHECKPOINT_PATH).stem
        viz_path = save_dir / f'{checkpoint_name}_codebook_usage.png'
        visualize_usage(usage_stats, viz_path)
        
        # 6. Salva JSON
        json_path = save_dir / f'{checkpoint_name}_codebook_usage.json'
        save_json_report(usage_stats, CHECKPOINT_PATH, json_path)
        
        print(f"\n✅ ANALISI COMPLETATA!")
        print(f"   Risultati salvati in: {save_dir}/")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())