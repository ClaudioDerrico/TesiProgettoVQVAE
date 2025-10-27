#!/usr/bin/env python3
"""
Script per verificare la Codebook Usage di un checkpoint VQ-VAE specifico.

Calcola:
- Codici usati / Codici totali (%)
- Distribuzione dell'uso dei codici
- Perplexity
- Statistiche dettagliate

Uso:
    1. Modifica CHECKPOINT_PATH nel main()
    2. python codebook_usage_checker.py
    
Configurazione:
    - CHECKPOINT_PATH: path al tuo file .pth
    - MAX_BATCHES: None (tutti) o numero per debug veloce
    - BATCH_SIZE: dimensione batch
    - SAVE_PLOT: None (auto) o path custom per salvare il grafico
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from models.vqvae import CalciumVQVAE
from datasets.calcium import create_unified_dataloaders, TRAINING_SESSION_IDS


def load_checkpoint_info(checkpoint_path):
    """Carica le informazioni dal checkpoint"""
    print(f"\n📂 Caricando checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Estrai configurazione modello
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print(f"✅ Configurazione trovata nel checkpoint")
    else:
        print(f"⚠️  Configurazione non trovata, uso default")
        config = None
    
    # Informazioni training
    if 'epoch' in checkpoint:
        print(f"📊 Epoca: {checkpoint['epoch']}")
    
    if 'best_correlation' in checkpoint:
        print(f"📊 Best correlation: {checkpoint['best_correlation']:.4f}")
    
    if 'training_sessions' in checkpoint:
        print(f"📊 Training sessions: {len(checkpoint['training_sessions'])}")
    
    return checkpoint, config


def create_model_from_config(config):
    """Crea il modello dalla configurazione"""
    
    if config is None:
        # Configurazione di fallback
        print("⚠️  Usando configurazione di fallback")
        config = {
            'num_neurons': 30,
            'num_hiddens': 512,
            'num_residual_layers': 6,
            'num_residual_hiddens': 256,
            'num_embeddings': 16,
            'embedding_dim': 32,
            'commitment_cost': 1.0,
            'dropout_rate': 0.0,
            'use_quantizer': True,
        }
    
    print(f"\n🏗️  Creando modello:")
    print(f"   Neurons: {config['num_neurons']}")
    print(f"   Embeddings: {config['num_embeddings']}")
    print(f"   Embedding dim: {config['embedding_dim']}")
    
    model = CalciumVQVAE(
        num_neurons=config['num_neurons'],
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        dropout_rate=config.get('dropout_rate', 0.0),
        use_quantizer=config.get('use_quantizer', True)
    )
    
    return model, config


def analyze_codebook_usage(model, dataloader, device, max_batches=None):
    """
    Analizza l'uso del codebook su un dataset.
    
    Returns:
        dict con statistiche di utilizzo
    """
    
    print(f"\n🔍 Analizzando utilizzo codebook...")
    
    model.eval()
    model = model.to(device)
    
    all_encoding_indices = []
    all_perplexities = []
    
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Gestisci diverse strutture di batch
            if isinstance(batch_data, (list, tuple)):
                neural_data = batch_data[0].to(device)
            else:
                neural_data = batch_data.to(device)
            
            # Forward pass
            vq_loss, recon, perplexity, encodings, encoding_indices = model(neural_data)
            
            # Salva indici
            all_encoding_indices.extend(encoding_indices.cpu().numpy().flatten().tolist())
            all_perplexities.append(perplexity.item())
            
            num_batches += 1
            
            # Progress
            if num_batches % 50 == 0:
                print(f"   Processati {num_batches} batch...")
            
            # Limite batch (per debug veloce)
            if max_batches and num_batches >= max_batches:
                print(f"   Fermato a {max_batches} batch (debug mode)")
                break
    
    print(f"✅ Analisi completata su {num_batches} batch")
    
    # ========================================================================
    # CALCOLA STATISTICHE
    # ========================================================================
    
    all_encoding_indices = np.array(all_encoding_indices)
    
    # Conta occorrenze di ogni codice
    code_counts = Counter(all_encoding_indices)
    
    num_embeddings = model.vector_quantization.n_e
    used_codes = len(code_counts)
    unused_codes = num_embeddings - used_codes
    
    # Codebook usage percentage
    usage_percentage = (used_codes / num_embeddings) * 100
    
    # Perplexity medio
    avg_perplexity = np.mean(all_perplexities)
    
    # Statistiche uso
    counts_array = np.array(list(code_counts.values()))
    
    stats = {
        'num_embeddings': num_embeddings,
        'used_codes': used_codes,
        'unused_codes': unused_codes,
        'usage_percentage': usage_percentage,
        'avg_perplexity': avg_perplexity,
        'total_assignments': len(all_encoding_indices),
        'code_counts': code_counts,
        'counts_array': counts_array,
        'all_encoding_indices': all_encoding_indices,
        
        # Statistiche distribuzione
        'min_usage': counts_array.min(),
        'max_usage': counts_array.max(),
        'mean_usage': counts_array.mean(),
        'std_usage': counts_array.std(),
        'median_usage': np.median(counts_array),
    }
    
    return stats


def print_usage_report(stats):
    """Stampa report dettagliato dell'utilizzo"""
    
    print("\n" + "="*70)
    print("📊 CODEBOOK USAGE REPORT")
    print("="*70)
    
    print(f"\n🎯 UTILIZZO CODEBOOK:")
    print(f"   Codici totali:     {stats['num_embeddings']}")
    print(f"   Codici usati:      {stats['used_codes']}")
    print(f"   Codici NON usati:  {stats['unused_codes']}")
    print(f"   Usage percentage:  {stats['usage_percentage']:.2f}%")
    
    print(f"\n📈 PERPLEXITY:")
    print(f"   Perplexity media:  {stats['avg_perplexity']:.2f}")
    print(f"   (Teorico max:      {stats['num_embeddings']})")
    
    print(f"\n📊 DISTRIBUZIONE USO CODICI:")
    print(f"   Total assignments: {stats['total_assignments']}")
    print(f"   Min usage:         {stats['min_usage']}")
    print(f"   Max usage:         {stats['max_usage']}")
    print(f"   Mean usage:        {stats['mean_usage']:.0f}")
    print(f"   Std usage:         {stats['std_usage']:.0f}")
    print(f"   Median usage:      {stats['median_usage']:.0f}")
    
    # Top 10 codici più usati
    print(f"\n🔝 TOP 10 CODICI PIÙ USATI:")
    top_10 = stats['code_counts'].most_common(10)
    for rank, (code_idx, count) in enumerate(top_10, 1):
        percentage = (count / stats['total_assignments']) * 100
        print(f"   {rank:2d}. Code {int(code_idx):3d}: {int(count):10d} volte ({percentage:5.2f}%)")
    
    # Codici non usati
    if stats['unused_codes'] > 0:
        print(f"\n⚠️  CODICI NON USATI ({stats['unused_codes']}):")
        
        all_codes = set(range(stats['num_embeddings']))
        used_codes_set = set(stats['code_counts'].keys())
        unused_codes_list = sorted(list(all_codes - used_codes_set))
        
        # Mostra primi 20 se sono tanti
        if len(unused_codes_list) > 20:
            print(f"   {unused_codes_list[:20]}... (+ altri {len(unused_codes_list) - 20})")
        else:
            print(f"   {unused_codes_list}")
    
    print("\n" + "="*70)


def plot_usage_distribution(stats, save_path=None):
    """Visualizza la distribuzione dell'uso dei codici"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Codebook Usage Analysis ({stats["usage_percentage"]:.1f}% used)', 
                 fontsize=16, fontweight='bold')
    
    # ========================================================================
    # 1. BAR CHART: Uso per codice
    # ========================================================================
    code_indices = np.arange(stats['num_embeddings'])
    usage_counts = np.zeros(stats['num_embeddings'])
    
    for code_idx, count in stats['code_counts'].items():
        usage_counts[code_idx] = count
    
    colors = ['red' if count == 0 else 'skyblue' for count in usage_counts]
    
    axes[0, 0].bar(code_indices, usage_counts, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Code Index')
    axes[0, 0].set_ylabel('Usage Count')
    axes[0, 0].set_title(f'Usage per Code ({stats["used_codes"]}/{stats["num_embeddings"]} used)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Evidenzia codici non usati
    if stats['unused_codes'] > 0:
        unused_label = f"Unused ({stats['unused_codes']})"
        axes[0, 0].bar([], [], color='red', alpha=0.7, label=unused_label)
        axes[0, 0].legend()
    
    # ========================================================================
    # 2. HISTOGRAM: Distribuzione frequenze
    # ========================================================================
    axes[0, 1].hist(stats['counts_array'], bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(stats['mean_usage'], color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {stats["mean_usage"]:.0f}')
    axes[0, 1].axvline(stats['median_usage'], color='orange', linestyle='--', 
                       linewidth=2, label=f'Median: {stats["median_usage"]:.0f}')
    axes[0, 1].set_xlabel('Usage Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Usage Counts')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # 3. PIE CHART: Used vs Unused
    # ========================================================================
    labels = [f'Used\n({stats["used_codes"]})', 
              f'Unused\n({stats["unused_codes"]})']
    sizes = [stats['used_codes'], stats['unused_codes']]
    colors_pie = ['skyblue', 'lightcoral']
    explode = (0.05, 0.05)
    
    axes[1, 0].pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                   autopct='%1.1f%%', shadow=True, startangle=90,
                   textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1, 0].set_title(f'Codebook Usage Overview\n({stats["usage_percentage"]:.1f}% used)')
    
    # ========================================================================
    # 4. CUMULATIVE: Top codes contribution
    # ========================================================================
    # Ordina codici per uso decrescente
    sorted_counts = np.sort(stats['counts_array'])[::-1]
    cumulative = np.cumsum(sorted_counts)
    cumulative_pct = (cumulative / stats['total_assignments']) * 100
    
    axes[1, 1].plot(range(1, len(cumulative_pct) + 1), cumulative_pct, 
                    'b-', linewidth=2)
    axes[1, 1].axhline(50, color='red', linestyle='--', 
                       linewidth=2, alpha=0.7, label='50%')
    axes[1, 1].axhline(80, color='orange', linestyle='--', 
                       linewidth=2, alpha=0.7, label='80%')
    axes[1, 1].axhline(95, color='green', linestyle='--', 
                       linewidth=2, alpha=0.7, label='95%')
    
    # Trova quanti codici servono per 50%, 80%, 95%
    codes_for_50 = np.argmax(cumulative_pct >= 50) + 1
    codes_for_80 = np.argmax(cumulative_pct >= 80) + 1
    codes_for_95 = np.argmax(cumulative_pct >= 95) + 1
    
    axes[1, 1].scatter([codes_for_50], [50], color='red', s=100, zorder=5)
    axes[1, 1].scatter([codes_for_80], [80], color='orange', s=100, zorder=5)
    axes[1, 1].scatter([codes_for_95], [95], color='green', s=100, zorder=5)
    
    axes[1, 1].set_xlabel('Number of Top Codes')
    axes[1, 1].set_ylabel('Cumulative Coverage (%)')
    axes[1, 1].set_title('Cumulative Coverage by Top Codes')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Annotazioni
    axes[1, 1].text(0.98, 0.05, 
                    f'50%: {codes_for_50} codes\n'
                    f'80%: {codes_for_80} codes\n'
                    f'95%: {codes_for_95} codes',
                    transform=axes[1, 1].transAxes,
                    fontsize=10,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Grafico salvato: {save_path}")
    
    plt.show()
    plt.close()


def main():
    # ========================================================================
    # ✅ CONFIGURA QUI IL TUO CHECKPOINT
    # ========================================================================

    CHECKPOINT_PATH = "results/best_single_neuron_model_1024_pt2.pth"  # ← CAMBIA QUI!

    MAX_BATCHES = None     # None = analizza tutto, oppure es: 100 per debug veloce
    BATCH_SIZE = 32        # Batch size per dataloader
    SAVE_PLOT = None       # None = auto-genera, oppure es: "my_plot.png"
    
    # ========================================================================
    
    # Verifica che il checkpoint esista
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint non trovato: {checkpoint_path}")
        print(f"   Assicurati che il path sia corretto!")
        return
    
    print("="*70)
    print("🔍 CODEBOOK USAGE CHECKER")
    print("="*70)
    print(f"📂 Checkpoint: {CHECKPOINT_PATH}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    # ========================================================================
    # 1. CARICA CHECKPOINT
    # ========================================================================
    checkpoint, config = load_checkpoint_info(checkpoint_path)
    
    # ========================================================================
    # 2. CREA MODELLO
    # ========================================================================
    model, config = create_model_from_config(config)
    
    # Carica pesi
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Pesi caricati")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Pesi caricati (formato semplice)")
    
    # ========================================================================
    # 3. CARICA DATASET
    # ========================================================================
    print(f"\n📊 Caricando dataset per analisi...")
    
    try:
        train_loader, test_loader, dataset_info = create_unified_dataloaders(
            train_session_ids=TRAINING_SESSION_IDS[:3],  # Solo 3 sessioni per velocità
            batch_size=BATCH_SIZE,
            test_split=0.2,
            window_size=60,
            stride=50,
            min_neurons=config['num_neurons'],
            num_workers=0,
            use_cross_session_test=False
        )
        
        print(f"✅ Dataset caricato: {dataset_info['train_samples']} train samples")
        
    except Exception as e:
        print(f"⚠️  Errore caricamento dataset: {e}")
        print(f"   Usando dataset ridotto...")
        
        # Fallback: usa solo una sessione
        from datasets.calcium import SimpleAllenBrainDataset
        from torch.utils.data import DataLoader
        
        dataset = SimpleAllenBrainDataset(
            window_size=60,
            stride=50,
            min_neurons=config['num_neurons'],
            session_id=TRAINING_SESSION_IDS[0],
            use_neuron_sliding=False
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
    
    # ========================================================================
    # 4. ANALIZZA CODEBOOK USAGE
    # ========================================================================
    stats = analyze_codebook_usage(
        model, 
        train_loader, 
        device,
        max_batches=MAX_BATCHES
    )
    
    # ========================================================================
    # 5. REPORT
    # ========================================================================
    print_usage_report(stats)
    
    # ========================================================================
    # 6. VISUALIZZA
    # ========================================================================
    save_path = SAVE_PLOT
    if save_path is None:
        # Auto-genera nome basato su checkpoint
        checkpoint_name = checkpoint_path.stem
        save_path = checkpoint_path.parent / f"{checkpoint_name}_codebook_usage.png"
    
    plot_usage_distribution(stats, save_path)
    
    print("\n" + "="*70)
    print("✅ ANALISI COMPLETATA!")
    print("="*70)


if __name__ == "__main__":
    main()