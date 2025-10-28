#!/usr/bin/env python3
"""
Analisi Utilizzo Codebook VQ-VAE - Single Neuron

Analizza quali codici del codebook vengono effettivamente utilizzati
durante l'encoding di un dataset per modelli SINGLE NEURON.

Uso:
    python scripts/codebook_usage_analysis.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import json
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from collections import Counter

from models.vqvae import CalciumVQVAE
from datasets.calcium import (
    UnifiedMultiSessionDataset,
    TRAINING_SESSION_IDS,
    TEST_SESSION_IDS
)

# ============================================================================
# CONFIGURAZIONE - MODIFICA QUI IL CHECKPOINT
# ============================================================================

# ✅ INSERISCI QUI IL PATH AL TUO CHECKPOINT
CHECKPOINT_PATH = "results/best_single_neuron_model_2048.pth"

# Configurazione modello single neuron
MODEL_CONFIG = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 2048,
    'embedding_dim': 128,
    'commitment_cost': 0.25,
    'dropout_rate': 0.1,
    'use_quantizer': True,
}

# Parametri analisi
DATASET_TO_ANALYZE = 'train'  # 'train', 'test', o 'both'
MAX_SAMPLES = None  # None = tutti i campioni
BATCH_SIZE = 64
SAVE_DIR = 'codebook_analysis_single_neuron'
USE_WANDB = False  # True per loggare su W&B

# ============================================================================
# FUNZIONI
# ============================================================================

def load_model(checkpoint_path, config, device):
    """Carica il modello dai pesi salvati"""
    print(f"🔄 Caricando modello da: {checkpoint_path}")
    
    model = CalciumVQVAE(
        num_neurons=config['num_neurons'],
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        dropout_rate=config['dropout_rate'],
        use_quantizer=config['use_quantizer']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Modello caricato (epoca {checkpoint.get('epoch', 'N/A')})")
        
        if 'training_sessions' in checkpoint:
            print(f"📊 Allenato su {len(checkpoint['training_sessions'])} sessioni")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Modello caricato")
    
    model = model.to(device)
    model.eval()
    
    return model


def analyze_codebook_usage(model, dataloader, device, max_samples=None):
    """
    Analizza l'utilizzo del codebook su un dataset.
    
    Returns:
        dict con statistiche di utilizzo
    """
    print(f"\n{'='*70}")
    print(f"📊 ANALISI UTILIZZO CODEBOOK")
    print(f"{'='*70}")
    
    num_embeddings = model.vector_quantization.n_e
    print(f"   Codebook size: {num_embeddings} codici")
    
    # Counter per tenere traccia dei codici usati
    code_usage = Counter()
    total_encodings = 0
    
    model.eval()
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Gestisci batch con o senza maschere
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 3:
                neural_data = batch_data[0].to(device)
            else:
                neural_data = batch_data.to(device)
            
            # Encode
            z = model.encoder(neural_data)
            z = model.pre_quantization_conv(z)
            
            # Quantize e ottieni gli indici
            _, _, _, _, encoding_indices = model.vector_quantization(z)
            
            # Conta i codici utilizzati
            indices_cpu = encoding_indices.cpu().numpy().flatten()
            code_usage.update(indices_cpu)
            total_encodings += len(indices_cpu)
            
            samples_processed += neural_data.shape[0]
            
            # Progress
            if (batch_idx + 1) % 50 == 0:
                print(f"   Processati {batch_idx + 1} batch, {samples_processed} campioni...")
            
            # Break se limite raggiunto
            if max_samples and samples_processed >= max_samples:
                break
    
    # Calcola statistiche
    used_codes = len(code_usage)
    usage_percentage = (used_codes / num_embeddings) * 100
    
    # Frequenze normalizzate
    code_frequencies = {}
    for code_idx in range(num_embeddings):
        count = code_usage.get(code_idx, 0)
        code_frequencies[code_idx] = {
            'count': count,
            'percentage': (count / total_encodings * 100) if total_encodings > 0 else 0
        }
    
    # Top 10 codici più usati
    top_10_codes = code_usage.most_common(10)
    
    # Codici mai usati
    unused_codes = [i for i in range(num_embeddings) if code_usage[i] == 0]
    
    print(f"\n✅ Analisi completata:")
    print(f"   Campioni processati: {samples_processed}")
    print(f"   Total encodings: {total_encodings:,}")
    print(f"   Codici usati: {used_codes}/{num_embeddings}")
    print(f"   Utilizzo: {usage_percentage:.2f}%")
    print(f"   Codici inutilizzati: {len(unused_codes)}")
    
    return {
        'num_embeddings': num_embeddings,
        'used_codes': used_codes,
        'usage_percentage': usage_percentage,
        'unused_codes': unused_codes,
        'code_frequencies': code_frequencies,
        'top_10_codes': top_10_codes,
        'total_encodings': total_encodings,
        'samples_processed': samples_processed,
        'code_usage_counter': code_usage
    }


def save_results_to_json(results, checkpoint_path, model_config, dataset_type):
    """
    Salva i risultati dell'analisi in formato JSON nella cartella code_usage.
    
    Args:
        results: Dizionario con le statistiche dell'analisi
        checkpoint_path: Path del checkpoint analizzato
        model_config: Configurazione del modello
        dataset_type: Tipo di dataset analizzato ('train', 'test', 'both')
    """
    # Crea la cartella code_usage se non esiste
    output_dir = Path('code_usage')
    output_dir.mkdir(parents=True, exist_ok=True)

    
    # Prepara i dati da salvare
    json_data = {
        'metadata': {
            'checkpoint_path': str(checkpoint_path),
            'dataset_analyzed': dataset_type,
            'model_config': model_config
        },
        'results': {}
    }
    
    # Processa i risultati per ogni dataset
    for dataset_name, stats in results.items():
        # Converti Counter in dict normale per serializzazione JSON
        # Assicurati che le chiavi siano int standard, non numpy.int64
        code_usage_dict = {int(k): int(v) for k, v in stats['code_usage_counter'].items()}
        
        # Converti top_10_codes in formato serializzabile
        top_10_serializable = [
            {'code_index': int(code_idx), 'count': int(count)}
            for code_idx, count in stats['top_10_codes']
        ]
        
        # Converti code_frequencies mantenendo gli int come chiavi di stringa per JSON
        code_frequencies_serializable = {
            str(k): {
                'count': int(v['count']),
                'percentage': float(v['percentage'])
            }
            for k, v in stats['code_frequencies'].items()
        }
        
        json_data['results'][dataset_name] = {
            'num_embeddings': int(stats['num_embeddings']),
            'used_codes': int(stats['used_codes']),
            'usage_percentage': float(stats['usage_percentage']),
            'unused_codes': [int(x) for x in stats['unused_codes']],
            'total_encodings': int(stats['total_encodings']),
            'samples_processed': int(stats['samples_processed']),
            'code_usage': code_usage_dict,
            'top_10_codes': top_10_serializable,
            'code_frequencies': code_frequencies_serializable
        }
    
    # Estrai il nome del checkpoint senza estensione e path
    checkpoint_name = Path(checkpoint_path).stem  # es: "best_single_neuron_model_16"
    
    # Nome del file basato su checkpoint
    filename = f'{checkpoint_name}.json'
    output_path = output_dir / filename
    
    # Salva il JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Risultati salvati in JSON:")
    print(f"   {output_path}")
    
    return output_path


def visualize_codebook_usage(usage_stats, save_dir):
    """Crea visualizzazioni dell'utilizzo del codebook"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_embeddings = usage_stats['num_embeddings']
    code_frequencies = usage_stats['code_frequencies']
    
    # ========================================================================
    # FIGURA 1: Overview utilizzo
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'📚 Codebook Usage Analysis\n'
                 f'{usage_stats["used_codes"]}/{num_embeddings} codes used '
                 f'({usage_stats["usage_percentage"]:.1f}%)',
                 fontsize=16, fontweight='bold')
    
    # 1. Bar chart: Frequenza di ogni codice
    code_indices = list(range(num_embeddings))
    frequencies_pct = [code_frequencies[i]['percentage'] for i in code_indices]
    
    colors = ['green' if f > 0 else 'red' for f in frequencies_pct]
    
    axes[0, 0].bar(code_indices, frequencies_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Code Index', fontsize=11)
    axes[0, 0].set_ylabel('Usage (%)', fontsize=11)
    axes[0, 0].set_title('Code Usage Frequency', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Linea media
    mean_usage = np.mean(frequencies_pct)
    axes[0, 0].axhline(mean_usage, color='blue', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_usage:.3f}%')
    axes[0, 0].legend(loc='upper right')
    
    # 2. Istogramma: Distribuzione frequenze
    non_zero_frequencies = [f for f in frequencies_pct if f > 0]
    
    if non_zero_frequencies:
        axes[0, 1].hist(non_zero_frequencies, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Usage (%)', fontsize=11)
        axes[0, 1].set_ylabel('Number of Codes', fontsize=11)
        axes[0, 1].set_title('Distribution of Non-Zero Usage', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Statistiche
        mean_non_zero = np.mean(non_zero_frequencies)
        median_non_zero = np.median(non_zero_frequencies)
        axes[0, 1].axvline(mean_non_zero, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_non_zero:.3f}%')
        axes[0, 1].axvline(median_non_zero, color='orange', linestyle='--', linewidth=2,
                          label=f'Median: {median_non_zero:.3f}%')
        axes[0, 1].legend(loc='upper right')
    else:
        axes[0, 1].text(0.5, 0.5, 'No codes used!', 
                       ha='center', va='center', fontsize=14, color='red')
        axes[0, 1].set_title('Distribution of Non-Zero Usage', fontsize=12, fontweight='bold')
    
    # 3. Pie chart: Used vs Unused
    used = usage_stats['used_codes']
    unused = len(usage_stats['unused_codes'])
    
    pie_colors = ['#66c2a5', '#fc8d62']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = axes[1, 0].pie(
        [used, unused],
        labels=['Used Codes', 'Unused Codes'],
        autopct='%1.1f%%',
        startangle=90,
        colors=pie_colors,
        explode=explode,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    axes[1, 0].set_title(f'Codebook Utilization\n({used}/{num_embeddings} codes used)', 
                         fontsize=12, fontweight='bold')
    
    # 4. Top 10 codici più usati
    top_10 = usage_stats['top_10_codes'][:10]
    
    if top_10:
        codes = [f"Code {idx}" for idx, _ in top_10]
        counts = [count for _, count in top_10]
        
        y_pos = np.arange(len(codes))
        
        axes[1, 1].barh(y_pos, counts, color='steelblue', edgecolor='black', alpha=0.8)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(codes, fontsize=9)
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_xlabel('Usage Count', fontsize=11)
        axes[1, 1].set_title('Top 10 Most Used Codes', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # Aggiungi percentuale
        for i, (idx, count) in enumerate(top_10):
            pct = (count / usage_stats['total_encodings'] * 100)
            axes[1, 1].text(count, i, f' {pct:.2f}%', 
                          va='center', fontsize=9, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'No codes used!',
                       ha='center', va='center', fontsize=14, color='red')
        axes[1, 1].set_title('Top 10 Most Used Codes', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'codebook_usage_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Overview salvato: {save_dir / 'codebook_usage_overview.png'}")
    
    # ========================================================================
    # FIGURA 2: Heatmap
    # ========================================================================
    if num_embeddings <= 256:  # Crea heatmap solo per codebook non troppo grandi
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Crea griglia quadrata
        grid_size = int(np.ceil(np.sqrt(num_embeddings)))
        usage_matrix = np.zeros((grid_size, grid_size))
        
        for i in range(num_embeddings):
            row = i // grid_size
            col = i % grid_size
            usage_matrix[row, col] = frequencies_pct[i]
        
        im = ax.imshow(usage_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        ax.set_title(f'Codebook Usage Heatmap ({num_embeddings} codes)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Grid Column', fontsize=11)
        ax.set_ylabel('Grid Row', fontsize=11)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Usage (%)')
        
        # Annota i codici
        for i in range(num_embeddings):
            row = i // grid_size
            col = i % grid_size
            
            text_color = 'white' if frequencies_pct[i] > 0.5 else 'black'
            ax.text(col, row, str(i), ha='center', va='center',
                   fontsize=8, color=text_color, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'codebook_usage_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Heatmap salvato: {save_dir / 'codebook_usage_heatmap.png'}")


def log_to_wandb(usage_stats, model_name):
    """Log statistiche su W&B"""
    
    wandb.log({
        f'{model_name}/codebook_size': usage_stats['num_embeddings'],
        f'{model_name}/codes_used': usage_stats['used_codes'],
        f'{model_name}/usage_percentage': usage_stats['usage_percentage'],
        f'{model_name}/codes_unused': len(usage_stats['unused_codes']),
        f'{model_name}/total_encodings': usage_stats['total_encodings'],
        f'{model_name}/samples_processed': usage_stats['samples_processed'],
    })
    
    # Histogram delle frequenze
    code_frequencies = usage_stats['code_frequencies']
    frequencies_pct = [code_frequencies[i]['percentage'] for i in range(usage_stats['num_embeddings'])]
    non_zero_freq = [f for f in frequencies_pct if f > 0]
    
    if non_zero_freq:
        wandb.log({
            f'{model_name}/frequency_distribution': wandb.Histogram(non_zero_freq)
        })
    
    # Top 10 table
    top_10 = usage_stats['top_10_codes']
    
    if top_10:
        table_data = []
        for code_idx, count in top_10:
            pct = (count / usage_stats['total_encodings'] * 100)
            table_data.append([code_idx, count, pct])
        
        table = wandb.Table(
            columns=["Code Index", "Count", "Usage (%)"],
            data=table_data
        )
        
        wandb.log({f'{model_name}/top_10_codes': table})


def create_dataset(session_ids, num_neurons, window_size, stride, mode='train', batch_size=32):
    """Crea dataset per analisi"""
    
    dataset = UnifiedMultiSessionDataset(
        session_ids=session_ids,
        window_size=window_size,
        stride=stride,
        min_neurons=num_neurons,
        mode=mode
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return loader, dataset


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("📚 ANALISI UTILIZZO CODEBOOK - SINGLE NEURON")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Codebook size: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Dataset: {DATASET_TO_ANALYZE}")
    print(f"   Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Save dir: {SAVE_DIR}")
    print(f"   Use W&B: {USE_WANDB}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Verifica esistenza checkpoint
    if not Path(CHECKPOINT_PATH).exists():
        print(f"\n❌ ERRORE: Checkpoint non trovato: {CHECKPOINT_PATH}")
        print(f"   Modifica la variabile CHECKPOINT_PATH nello script!")
        return
    
    # Inizializza W&B (opzionale)
    if USE_WANDB:
        wandb.init(
            project="calcium-vqvae-codebook-analysis",
            name=f"codebook-usage-single-neuron-{datetime.now().strftime('%m%d-%H%M')}",
            config={
                "checkpoint": CHECKPOINT_PATH,
                "model_type": "single_neuron",
                "dataset": DATASET_TO_ANALYZE,
                "max_samples": MAX_SAMPLES,
                **MODEL_CONFIG
            },
            tags=["codebook-analysis", "single-neuron"]
        )
        print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # Carica modello
        model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # Parametri dataset
        window_size = 60
        stride = 50
        num_neurons = MODEL_CONFIG['num_neurons']
        
        # Analizza dataset(s)
        results = {}
        
        if DATASET_TO_ANALYZE in ['train', 'both']:
            print("\n" + "="*70)
            print("🔴 ANALISI TRAINING DATASET")
            print("="*70)
            
            train_loader, _ = create_dataset(
                TRAINING_SESSION_IDS, num_neurons, window_size, stride,
                mode='train', batch_size=BATCH_SIZE
            )
            
            train_stats = analyze_codebook_usage(
                model, train_loader, device, max_samples=MAX_SAMPLES
            )
            
            results['train'] = train_stats
            
            # Visualizza
            save_dir_train = Path(SAVE_DIR) / 'train'
            visualize_codebook_usage(train_stats, save_dir_train)
            
            if USE_WANDB:
                log_to_wandb(train_stats, 'train')
        
        if DATASET_TO_ANALYZE in ['test', 'both']:
            print("\n" + "="*70)
            print("🔵 ANALISI TEST DATASET")
            print("="*70)
            
            test_loader, _ = create_dataset(
                TEST_SESSION_IDS, num_neurons, window_size, stride,
                mode='test', batch_size=BATCH_SIZE
            )
            
            test_stats = analyze_codebook_usage(
                model, test_loader, device, max_samples=MAX_SAMPLES
            )
            
            results['test'] = test_stats
            
            # Visualizza
            save_dir_test = Path(SAVE_DIR) / 'test'
            visualize_codebook_usage(test_stats, save_dir_test)
            
            if USE_WANDB:
                log_to_wandb(test_stats, 'test')
        
        # 💾 SALVA I RISULTATI IN JSON
        json_path = save_results_to_json(
            results, 
            CHECKPOINT_PATH, 
            MODEL_CONFIG, 
            DATASET_TO_ANALYZE
        )
        
        # Summary finale
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        
        for dataset_name, stats in results.items():
            print(f"\n{dataset_name.upper()}:")
            print(f"   Codebook size: {stats['num_embeddings']}")
            print(f"   Codes used: {stats['used_codes']} ({stats['usage_percentage']:.2f}%)")
            print(f"   Codes unused: {len(stats['unused_codes'])}")
            print(f"   Total encodings: {stats['total_encodings']:,}")
            
            if stats['top_10_codes']:
                print(f"\n   Top 5 most used codes:")
                for i, (code_idx, count) in enumerate(stats['top_10_codes'][:5], 1):
                    pct = (count / stats['total_encodings'] * 100)
                    print(f"      {i}. Code {code_idx}: {count:,} times ({pct:.2f}%)")
        
        print(f"\n✅ Risultati salvati in: {SAVE_DIR}/")
        print(f"✅ Risultati JSON salvati in: code_usage/")
        
        if USE_WANDB:
            print(f"🌐 Risultati W&B: {wandb.run.url}")
        
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if USE_WANDB:
            wandb.finish()
            print("✅ W&B run completato!")


if __name__ == "__main__":
    main()