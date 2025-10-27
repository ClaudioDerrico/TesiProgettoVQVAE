#!/usr/bin/env python3
"""
Analisi Utilizzo Codebook VQ-VAE - SINGLE NEURON (FIXED)

✅ FIX 1: JSON serialization error (numpy types → Python types)
✅ FIX 2: Report semplificato (solo statistiche, no lista completa codici)

Questo script analizza quali codici del codebook vengono effettivamente
utilizzati durante la ricostruzione per modelli VQ-VAE single neuron.

Output semplificato:
    ✅ Codici usati: 127 / 1024 (12.40%)
    📊 Top 10 codici più usati
    📊 Statistiche distribuzione

🔧 COME USARE:
    1. Modifica CHECKPOINT_PATH
    2. Modifica MODEL_CONFIG
    3. python analyze_codebook_usage_FIXED.py
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
# CONFIGURAZIONE - MODIFICA QUI!
# ============================================================================

CHECKPOINT_PATH = "results/best_single_neuron_model_128.pth"

MODEL_CONFIG = {
    'num_neurons': 1,           
    'num_hiddens': 128,         
    'num_residual_layers': 3,   
    'num_residual_hiddens': 64, 
    'num_embeddings': 128,
    'embedding_dim': 32,
    'commitment_cost': 0.5,       
    'dropout_rate': 0.0,        
    'use_quantizer': True,
}

BATCH_SIZE = 64
NUM_SAMPLES = None  # None = tutto, oppure es. 500
SAVE_DIR = "codebook_analysis"

DEFAULT_WINDOW_SIZE = 60
DEFAULT_STRIDE = 50

# ============================================================================
# FUNZIONI
# ============================================================================

def load_single_neuron_model(checkpoint_path, device):
    """Carica modello SINGLE NEURON"""
    print(f"\n🔄 Caricando checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint non trovato: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    if 'vector_quantization.embedding.weight' in state_dict:
        num_embeddings = state_dict['vector_quantization.embedding.weight'].shape[0]
        embedding_dim_actual = state_dict['vector_quantization.embedding.weight'].shape[1]
        print(f"\n📊 Codebook rilevato:")
        print(f"   Num embeddings: {num_embeddings}")
        print(f"   Embedding dim: {embedding_dim_actual}")
    else:
        raise ValueError("❌ Codebook non trovato!")
    
    model = CalciumVQVAE(
        num_neurons=MODEL_CONFIG['num_neurons'],
        num_hiddens=MODEL_CONFIG['num_hiddens'],
        num_residual_layers=MODEL_CONFIG['num_residual_layers'],
        num_residual_hiddens=MODEL_CONFIG['num_residual_hiddens'],
        num_embeddings=num_embeddings,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        commitment_cost=MODEL_CONFIG['commitment_cost'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        use_quantizer=MODEL_CONFIG['use_quantizer']
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modello caricato")
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'best_correlation': checkpoint.get('best_correlation', None),
    }
    
    return model, num_embeddings, checkpoint_info


def analyze_codebook_usage(model, dataloader, device, num_samples=None):
    """Analizza utilizzo codebook"""
    print(f"\n🔍 Analizzando utilizzo codebook...")
    
    model.eval()
    code_counter = Counter()
    
    total_samples = 0
    total_timesteps = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            neural_data = batch_data.to(device)
            batch_size = neural_data.shape[0]
            
            vq_loss, recon, perplexity, encodings, encoding_indices = model(neural_data)
            
            if encoding_indices is not None:
                indices_flat = encoding_indices.flatten().cpu().numpy()
                
                for idx in indices_flat:
                    code_counter[int(idx)] += 1
                
                total_timesteps += len(indices_flat)
            
            total_samples += batch_size
            
            if (batch_idx + 1) % 50 == 0:
                print(f"   Processati {batch_idx + 1} batch, {total_samples} samples...")
            
            if num_samples is not None and total_samples >= num_samples:
                print(f"   Raggiunto limite di {num_samples} samples")
                break
    
    print(f"\n✅ Analisi completata:")
    print(f"   Samples: {total_samples}")
    print(f"   Timesteps: {total_timesteps}")
    
    num_codes_total = model.vector_quantization.n_e
    num_codes_used = len(code_counter)
    usage_percentage = (num_codes_used / num_codes_total) * 100
    
    print(f"   Codici usati: {num_codes_used}/{num_codes_total} ({usage_percentage:.2f}%)")
    
    sorted_codes = sorted(code_counter.items(), key=lambda x: x[1], reverse=True)
    top_10 = sorted_codes[:10]
    bottom_used = sorted_codes[-10:] if len(sorted_codes) >= 10 else sorted_codes
    
    unused_codes = set(range(num_codes_total)) - set(code_counter.keys())
    num_unused = len(unused_codes)
    
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
        'num_codes_total': int(num_codes_total),  # ✅ Convert to Python int
        'num_codes_used': int(num_codes_used),
        'num_codes_unused': int(num_unused),
        'usage_percentage': float(usage_percentage),
        'code_counter': {int(k): int(v) for k, v in code_counter.items()},  # ✅ Convert keys/values
        'sorted_codes': [(int(c), int(cnt)) for c, cnt in sorted_codes],
        'top_10': [(int(c), int(cnt)) for c, cnt in top_10],
        'bottom_used': [(int(c), int(cnt)) for c, cnt in bottom_used],
        'unused_codes': sorted([int(x) for x in unused_codes]),
        'total_samples': int(total_samples),
        'total_timesteps': int(total_timesteps),
        'mean_usage': float(mean_usage),
        'std_usage': float(std_usage),
        'median_usage': float(median_usage),
        'max_usage': int(max_usage),
        'min_usage': int(min_usage),
    }
    
    return usage_stats


def print_usage_report(usage_stats, checkpoint_path):
    """
    ✅ REPORT SEMPLIFICATO - Solo statistiche aggregate
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
    
    # Statistiche distribuzione
    if usage_stats['num_codes_used'] > 0:
        print(f"\n📊 STATISTICHE DISTRIBUZIONE:")
        print(f"   Mean usage: {usage_stats['mean_usage']:.1f}")
        print(f"   Median usage: {usage_stats['median_usage']:.1f}")
        print(f"   Std usage: {usage_stats['std_usage']:.1f}")
        print(f"   Max usage: {usage_stats['max_usage']}")
        print(f"   Min usage (escluso 0): {usage_stats['min_usage']}")
        
        # Top 10 codici più usati
        print(f"\n🔝 TOP 10 CODICI PIÙ USATI:")
        for i, (code_idx, count) in enumerate(usage_stats['top_10'][:10], 1):
            percentage = (count / usage_stats['total_timesteps']) * 100
            print(f"   {i:2d}. Codice {code_idx:4d}: {count:8d} volte ({percentage:5.2f}%)")
    
    print("="*70)


def visualize_usage(usage_stats, save_path):
    """Crea visualizzazioni"""
    
    num_codes = usage_stats['num_codes_total']
    code_counter = usage_stats['code_counter']
    
    usage_array = np.array([code_counter.get(i, 0) for i in range(num_codes)])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Codebook Usage Analysis ({num_codes} codes)', fontsize=16, fontweight='bold')
    
    # Plot 1: Bar plot
    ax = axes[0, 0]
    colors = ['red' if count == 0 else 'skyblue' for count in usage_array]
    ax.bar(range(num_codes), usage_array, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Code Index')
    ax.set_ylabel('Usage Count')
    ax.set_title(f'Usage per Code (Red = Unused)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Histogram
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
    
    # Plot 3: Cumulative
    ax = axes[1, 0]
    sorted_usage = np.sort(usage_array)[::-1]
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
    
    # Plot 4: Pie chart
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
    """✅ Salva JSON MINIMALE - Solo codici usati e percentuale"""
    
    report = {
        'checkpoint': os.path.basename(checkpoint_path),
        'timestamp': datetime.now().isoformat(),
        'num_codes_total': usage_stats['num_codes_total'],
        'num_codes_used': usage_stats['num_codes_used'],
        'code_usage_percentage': usage_stats['usage_percentage']
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Report JSON salvato: {save_path}")
    print(f"   Contiene: codici totali, codici usati, percentuale utilizzo")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function"""
    
    print("="*70)
    print("🔬 ANALISI CODEBOOK USAGE - SINGLE NEURON (FIXED)")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Num samples: {NUM_SAMPLES if NUM_SAMPLES else 'Tutti'}")
    print(f"   Save dir: {SAVE_DIR}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Carica modello
        model, num_embeddings, checkpoint_info = load_single_neuron_model(
            CHECKPOINT_PATH, device
        )
        
        print(f"\n📋 Info checkpoint:")
        print(f"   Epoca: {checkpoint_info['epoch']}")
        if checkpoint_info['best_correlation'] is not None:
            print(f"   Best correlation: {checkpoint_info['best_correlation']:.4f}")
        
        # Carica dataset
        print(f"\n📊 Caricando dataset...")
        train_loader, test_loader, dataset_info = create_unified_dataloaders(
            train_session_ids=TRAINING_SESSION_IDS,
            batch_size=BATCH_SIZE,
            test_split=0.2,
            window_size=DEFAULT_WINDOW_SIZE,
            stride=DEFAULT_STRIDE,
            min_neurons=1,  # SINGLE NEURON
            num_workers=0,
            use_cross_session_test=False
        )
        
        print(f"✅ Dataset caricato:")
        print(f"   Train: {dataset_info['train_samples']}")
        print(f"   Test: {dataset_info['test_samples']}")
        
        # Analizza
        usage_stats = analyze_codebook_usage(
            model, train_loader, device, 
            num_samples=NUM_SAMPLES
        )
        
        # Report
        print_usage_report(usage_stats, CHECKPOINT_PATH)
        
        # Visualizza
        checkpoint_name = Path(CHECKPOINT_PATH).stem
        viz_path = save_dir / f'{checkpoint_name}_codebook_usage.png'
        visualize_usage(usage_stats, viz_path)
        
        # Salva JSON
        json_path = save_dir / f'{checkpoint_name}_codebook_usage.json'
        save_json_report(usage_stats, CHECKPOINT_PATH, json_path)
        
        print(f"\n✅ ANALISI COMPLETATA!")
        print(f"   Risultati in: {save_dir}/")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())