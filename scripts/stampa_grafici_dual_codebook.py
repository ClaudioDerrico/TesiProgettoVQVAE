#!/usr/bin/env python3
"""
Analisi Dual Codebook con Visualizzazioni Separate

Analizza i due codebook (active/inactive) separatamente:
1. Decodifica tutti i codici di entrambi i codebook
2. Analisi caratteristiche separate
3. UMAP 2D e 3D per ogni codebook
4. Confronto tra i due codebook
5. Dendrogrammi gerarchici separati e combinati

Uso:
    python scripts/stampa_grafici_dual_codebook.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("⚠️  UMAP not available")
    UMAP_AVAILABLE = False

from models.dual_vqvae import DualCalciumVQVAE
from datasets.calcium import create_simple_calcium_dataloaders
from collections import Counter

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# ⚙️ CONFIGURAZIONE: Modifica questi valori in base al modello che vuoi analizzare
CHECKPOINT_PATH = "results/dual_codebook_constrained/best_dual_model_32_constrained.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-dual-analysis-constrained"
WANDB_RUN_NAME = f"dual-analysis-constrained-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ⚙️ CONFIGURAZIONE MODELLO: Deve corrispondere al checkpoint caricato
MODEL_CONFIG = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 32,         # 32 total = 16 active + 16 inactive
    'embedding_dim': 32,          # Dimensione embedding per modello 32
    'commitment_cost': 0.25,
    'dropout_rate': 0.0,
    'use_quantizer': True,
    'threshold_type': 'adaptive',
}

UMAP_PARAMS = {
    'n_neighbors': 10,
    'min_dist': 0.1,
    'metric': 'cosine',
}

# ============================================================================
# FUNZIONI
# ============================================================================

def load_model(checkpoint_path, config, device):
    """Carica modello"""
    print(f"🔄 Caricando modello da: {checkpoint_path}")
    
    model = DualCalciumVQVAE(**config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Modello caricato (epoca {checkpoint.get('epoch', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Modello caricato")
    
    model = model.to(device)
    model.eval()
    
    return model


def analyze_codebook_usage(model, dataloader, device, max_samples=None):
    """
    Analizza l'utilizzo effettivo del dual codebook su un dataset.
    
    Returns:
        dict con:
        - used_active_indices: lista di indici utilizzati nel codebook active
        - used_inactive_indices: lista di indici utilizzati nel codebook inactive
        - usage_stats: statistiche di utilizzo
    """
    print(f"\n{'='*70}")
    print(f"📊 ANALISI UTILIZZO DUAL CODEBOOK")
    print(f"{'='*70}")
    
    vq = model.vector_quantization
    n_e_per_codebook = vq.n_e_per_codebook
    
    # Contatori per tracciare quali codici vengono usati
    active_usage = Counter()
    inactive_usage = Counter()
    total_encodings = 0
    
    model.eval()
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Gestisci batch con o senza maschere
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
                neural_data = batch_data[0].to(device)
            else:
                neural_data = batch_data.to(device)
            
            # Forward pass
            z = model.encoder(neural_data)
            z = model.pre_quantization_conv(z)
            
            # Quantize e ottieni gli indici
            _, _, _, _, encoding_indices = vq(z)
            
            # Gli indici sono già offset: 0..n_e_per_codebook-1 = active, n_e_per_codebook..n_e-1 = inactive
            indices_cpu = encoding_indices.cpu().numpy().flatten()
            
            # Separa active e inactive
            active_mask = indices_cpu < n_e_per_codebook
            inactive_mask = indices_cpu >= n_e_per_codebook
            
            # Indici nel rispettivo codebook (rimuovi offset per inactive)
            active_indices = indices_cpu[active_mask]
            inactive_indices = indices_cpu[inactive_mask] - n_e_per_codebook
            
            active_usage.update(active_indices)
            inactive_usage.update(inactive_indices)
            total_encodings += len(indices_cpu)
            
            samples_processed += neural_data.shape[0]
            
            if (batch_idx + 1) % 50 == 0:
                print(f"   Processati {batch_idx + 1} batch, {samples_processed} campioni...")
            
            if max_samples and samples_processed >= max_samples:
                break
    
    # Trova indici utilizzati
    used_active_indices = sorted(list(active_usage.keys()))
    used_inactive_indices = sorted(list(inactive_usage.keys()))
    
    # Statistiche
    active_usage_pct = (len(used_active_indices) / n_e_per_codebook) * 100
    inactive_usage_pct = (len(used_inactive_indices) / n_e_per_codebook) * 100
    total_usage_pct = ((len(used_active_indices) + len(used_inactive_indices)) / (n_e_per_codebook * 2)) * 100
    
    print(f"\n✅ Analisi completata:")
    print(f"   Campioni processati: {samples_processed}")
    print(f"   Total encodings: {total_encodings:,}")
    print(f"   Active codebook: {len(used_active_indices)}/{n_e_per_codebook} utilizzati ({active_usage_pct:.1f}%)")
    print(f"   Inactive codebook: {len(used_inactive_indices)}/{n_e_per_codebook} utilizzati ({inactive_usage_pct:.1f}%)")
    print(f"   Totale: {len(used_active_indices) + len(used_inactive_indices)}/{n_e_per_codebook * 2} utilizzati ({total_usage_pct:.1f}%)")
    
    return {
        'used_active_indices': used_active_indices,
        'used_inactive_indices': used_inactive_indices,
        'active_usage': active_usage,
        'inactive_usage': inactive_usage,
        'usage_stats': {
            'active_used': len(used_active_indices),
            'active_total': n_e_per_codebook,
            'active_usage_pct': active_usage_pct,
            'inactive_used': len(used_inactive_indices),
            'inactive_total': n_e_per_codebook,
            'inactive_usage_pct': inactive_usage_pct,
            'total_used': len(used_active_indices) + len(used_inactive_indices),
            'total_total': n_e_per_codebook * 2,
            'total_usage_pct': total_usage_pct,
        }
    }


def extract_separate_embeddings(model, usage_info=None):
    """
    Estrae embeddings dei due codebook separatamente.
    
    Se usage_info è fornito, filtra solo gli embeddings utilizzati.
    
    Args:
        model: modello dual VQ-VAE
        usage_info: dict da analyze_codebook_usage (opzionale)
    
    Returns:
        embeddings_active, embeddings_inactive, (indices_active, indices_inactive se filtrato)
    """
    vq = model.vector_quantization
    
    embeddings_active = vq.embedding_active.weight.data.cpu().numpy()
    embeddings_inactive = vq.embedding_inactive.weight.data.cpu().numpy()
    
    if usage_info is not None:
        # Filtra solo embeddings utilizzati
        used_active_idx = usage_info['used_active_indices']
        used_inactive_idx = usage_info['used_inactive_indices']
        
        embeddings_active = embeddings_active[used_active_idx]
        embeddings_inactive = embeddings_inactive[used_inactive_idx]
        
        print(f"✅ Embeddings estratti (FILTRATI per utilizzo):")
        print(f"   Active codebook: {embeddings_active.shape} (da {len(used_active_idx)} codici utilizzati)")
        print(f"   Inactive codebook: {embeddings_inactive.shape} (da {len(used_inactive_idx)} codici utilizzati)")
        
        return embeddings_active, embeddings_inactive, (used_active_idx, used_inactive_idx)
    else:
        print(f"✅ Embeddings estratti (TUTTI):")
        print(f"   Active codebook: {embeddings_active.shape}")
        print(f"   Inactive codebook: {embeddings_inactive.shape}")
        
        return embeddings_active, embeddings_inactive, None


def decode_codebook(model, device, codebook_type='active', temporal_length=15, used_indices=None):
    """
    Decodifica un codebook (o solo i codici utilizzati se specified)
    
    Args:
        codebook_type: 'active' o 'inactive'
        used_indices: lista di indici da decodificare (None = tutti)
    """
    vq = model.vector_quantization
    
    if codebook_type == 'active':
        embeddings = vq.embedding_active.weight.data
        n_codes = vq.n_e_per_codebook
    else:
        embeddings = vq.embedding_inactive.weight.data
        n_codes = vq.n_e_per_codebook
    
    if used_indices is not None:
        indices_to_decode = used_indices
        print(f"\n🔍 Decodificando codebook {codebook_type.upper()} ({len(indices_to_decode)} codici utilizzati su {n_codes})...")
    else:
        indices_to_decode = list(range(n_codes))
        print(f"\n🔍 Decodificando codebook {codebook_type.upper()} ({n_codes} codici)...")
    
    decoded = []
    
    model.eval()
    with torch.no_grad():
        for idx, i in enumerate(indices_to_decode):
            code_embedding = embeddings[i]
            sequence = code_embedding.unsqueeze(1).repeat(1, temporal_length)
            sequence = sequence.unsqueeze(0).to(device)
            
            reconstruction = model.decode(sequence)
            decoded.append(reconstruction.cpu().numpy()[0])
            
            if (idx + 1) % 50 == 0:
                print(f"   Processati {idx+1}/{len(indices_to_decode)}...")
    
    decoded = np.array(decoded)
    print(f"✅ Decodifica completata: {decoded.shape}")
    
    return decoded


def analyze_characteristics(decoded_codes):
    """Analizza caratteristiche dei codici"""
    n_codes = decoded_codes.shape[0]
    
    chars = {
        'amplitude': [],
        'mean': [],
        'variance': [],
        'energy': [],
        'max_value': [],
        'min_value': [],
    }
    
    for i in range(n_codes):
        trace = decoded_codes[i, 0, :]  # Primo neurone
        
        chars['amplitude'].append(np.max(trace) - np.min(trace))
        chars['mean'].append(np.mean(trace))
        chars['variance'].append(np.var(trace))
        chars['energy'].append(np.sum(trace ** 2))
        chars['max_value'].append(np.max(trace))
        chars['min_value'].append(np.min(trace))
    
    for key in chars:
        chars[key] = np.array(chars[key])
    
    return chars


def perform_umap_on_codebook(embeddings, n_components=2):
    """UMAP su un singolo codebook"""
    if not UMAP_AVAILABLE:
        return None
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_PARAMS['n_neighbors'],
        min_dist=UMAP_PARAMS['min_dist'],
        metric=UMAP_PARAMS['metric'],
        random_state=42
    )
    
    coords = reducer.fit_transform(embeddings)
    return coords


def plot_dual_codebook_comparison(emb_active, emb_inactive, chars_active, chars_inactive):
    """
    Confronto visivo tra i due codebook
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🔬 Dual Codebook Comparison', fontsize=16, fontweight='bold')
    
    # 1. Amplitude distribution
    axes[0, 0].hist(chars_active['amplitude'], bins=30, alpha=0.6, label='Active', color='red')
    axes[0, 0].hist(chars_inactive['amplitude'], bins=30, alpha=0.6, label='Inactive', color='blue')
    axes[0, 0].set_xlabel('Amplitude')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Amplitude Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mean activity distribution
    axes[0, 1].hist(chars_active['mean'], bins=30, alpha=0.6, label='Active', color='red')
    axes[0, 1].hist(chars_inactive['mean'], bins=30, alpha=0.6, label='Inactive', color='blue')
    axes[0, 1].set_xlabel('Mean Activity')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Mean Activity Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Variance distribution
    axes[0, 2].hist(chars_active['variance'], bins=30, alpha=0.6, label='Active', color='red')
    axes[0, 2].hist(chars_inactive['variance'], bins=30, alpha=0.6, label='Inactive', color='blue')
    axes[0, 2].set_xlabel('Variance')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Variance Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Energy distribution
    axes[1, 0].hist(chars_active['energy'], bins=30, alpha=0.6, label='Active', color='red')
    axes[1, 0].hist(chars_inactive['energy'], bins=30, alpha=0.6, label='Inactive', color='blue')
    axes[1, 0].set_xlabel('Energy')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Energy Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Box plots comparison
    data_to_plot = [chars_active['amplitude'], chars_inactive['amplitude']]
    axes[1, 1].boxplot(data_to_plot, labels=['Active', 'Inactive'])
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Amplitude Boxplot')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    axes[1, 2].axis('off')
    stats_text = "Summary Statistics:\n\n"
    stats_text += "ACTIVE CODEBOOK:\n"
    stats_text += f"  Amplitude: {chars_active['amplitude'].mean():.3f} ± {chars_active['amplitude'].std():.3f}\n"
    stats_text += f"  Mean: {chars_active['mean'].mean():.3f} ± {chars_active['mean'].std():.3f}\n"
    stats_text += f"  Variance: {chars_active['variance'].mean():.3f} ± {chars_active['variance'].std():.3f}\n\n"
    stats_text += "INACTIVE CODEBOOK:\n"
    stats_text += f"  Amplitude: {chars_inactive['amplitude'].mean():.3f} ± {chars_inactive['amplitude'].std():.3f}\n"
    stats_text += f"  Mean: {chars_inactive['mean'].mean():.3f} ± {chars_inactive['mean'].std():.3f}\n"
    stats_text += f"  Variance: {chars_inactive['variance'].mean():.3f} ± {chars_inactive['variance'].std():.3f}\n"
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    return fig


def plot_umap_dual_comparison(umap_active, umap_inactive, chars_active, chars_inactive):
    """
    UMAP comparison tra i due codebook.
    Gestisce il caso in cui uno dei codebook sia vuoto.
    """
    # Determina quanti subplot servono
    n_plots = 0
    if umap_active is not None and len(umap_active) > 0:
        n_plots += 1
    if umap_inactive is not None and len(umap_inactive) > 0:
        n_plots += 1
    
    if n_plots == 0:
        # Nessun dato disponibile
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.text(0.5, 0.5, 'Nessun embedding disponibile per UMAP', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        return fig
    
    # Crea subplot appropriati
    if n_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    fig.suptitle('🎯 UMAP: Active vs Inactive Codebook', fontsize=16, fontweight='bold')
    
    plot_idx = 0
    
    # 1. Active codebook
    if umap_active is not None and len(umap_active) > 0:
        scatter1 = axes[plot_idx].scatter(umap_active[:, 0], umap_active[:, 1],
                                  c=chars_active['amplitude'], cmap='Reds',
                                  s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
        axes[plot_idx].set_title('Active Codebook', fontsize=13, fontweight='bold')
        axes[plot_idx].set_xlabel('UMAP 1')
        axes[plot_idx].set_ylabel('UMAP 2')
        plt.colorbar(scatter1, ax=axes[plot_idx], label='Amplitude')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # 2. Inactive codebook
    if umap_inactive is not None and len(umap_inactive) > 0:
        scatter2 = axes[plot_idx].scatter(umap_inactive[:, 0], umap_inactive[:, 1],
                                  c=chars_inactive['amplitude'], cmap='Blues',
                                  s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
        axes[plot_idx].set_title('Inactive Codebook', fontsize=13, fontweight='bold')
        axes[plot_idx].set_xlabel('UMAP 1')
        axes[plot_idx].set_ylabel('UMAP 2')
        plt.colorbar(scatter2, ax=axes[plot_idx], label='Amplitude')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # 3. Combined (solo se entrambi disponibili)
    if n_plots == 3:
        axes[2].scatter(umap_active[:, 0], umap_active[:, 1],
                       c='red', s=100, alpha=0.6, edgecolors='black', 
                       linewidths=0.5, label='Active')
        axes[2].scatter(umap_inactive[:, 0], umap_inactive[:, 1],
                       c='blue', s=100, alpha=0.6, edgecolors='black',
                       linewidths=0.5, label='Inactive')
        axes[2].set_title('Combined View', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('UMAP 1')
        axes[2].set_ylabel('UMAP 2')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_hierarchical_clustering_dual(emb_active, emb_inactive, usage_info=None):
    """
    Dendrogramma gerarchico combinato per i due codebook.
    
    Se usage_info è fornito, mostra solo gli embeddings utilizzati.
    
    Args:
        emb_active: embeddings del codebook active (già filtrati se usage_info fornito)
        emb_inactive: embeddings del codebook inactive (già filtrati se usage_info fornito)
        usage_info: dict da analyze_codebook_usage (opzionale)
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    
    # Determina se stiamo mostrando solo embeddings utilizzati
    if usage_info is not None:
        title_suffix = " (Solo Embeddings Utilizzati)"
        usage_stats = usage_info['usage_stats']
        info_text = f"Active: {usage_stats['active_used']}/{usage_stats['active_total']} | Inactive: {usage_stats['inactive_used']}/{usage_stats['inactive_total']}"
    else:
        title_suffix = " (Tutti gli Embeddings)"
        info_text = ""
    
    fig.suptitle(f'🌳 Hierarchical Clustering: Combined Dual Codebook{title_suffix}', 
                 fontsize=16, fontweight='bold')
    
    # Combina i due codebook
    combined_embeddings = np.vstack([emb_active, emb_inactive])
    n_active = len(emb_active)
    n_inactive = len(emb_inactive)
    
    if len(combined_embeddings) < 2:
        ax.text(0.5, 0.5, 'Troppo pochi embeddings per clustering', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        plt.tight_layout()
        return fig
    
    # Calcola distanze e linkage sul dataset combinato
    distances_combined = pdist(combined_embeddings, metric='cosine')
    linkage_combined = linkage(distances_combined, method='ward')
    
    # Crea il dendrogramma
    dendro = dendrogram(
        linkage_combined, 
        ax=ax, 
        color_threshold=0.5*max(linkage_combined[:,2]) if len(linkage_combined) > 0 else 0,
        no_labels=True  # Disabilita le etichette numeriche di default
    )
    
    title = f'Combined Codebook Dendrogram (Red: Active, Blue: Inactive)'
    if info_text:
        title += f'\n{info_text}'
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Code Index', fontsize=11)
    ax.set_ylabel('Distance', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Aggiungi linee verticali per separare visivamente i due codebook
    if n_active > 0:
        ax.axvline(x=n_active * 10, color='gray', linestyle='--', 
                   linewidth=2, alpha=0.5, label='Active/Inactive boundary')
        ax.legend()
    
    # Aggiungi annotazioni
    if n_active > 0:
        ax.text(0.25, 0.95, 'ACTIVE CODEBOOK', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                color='red', ha='center')
    if n_inactive > 0:
        ax.text(0.75, 0.95, 'INACTIVE CODEBOOK', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                color='blue', ha='center')
    
    plt.tight_layout()
    return fig

def plot_example_reconstructions(decoded_active, decoded_inactive, n_examples=5):
    """
    Plot esempi di ricostruzioni da entrambi i codebook
    """
    fig, axes = plt.subplots(2, n_examples, figsize=(20, 6))
    fig.suptitle('🎨 Example Reconstructions from Each Codebook', 
                 fontsize=16, fontweight='bold')
    
    # Random indices
    np.random.seed(42)
    active_indices = np.random.choice(len(decoded_active), n_examples, replace=False)
    inactive_indices = np.random.choice(len(decoded_inactive), n_examples, replace=False)
    
    # Active examples
    for i, idx in enumerate(active_indices):
        trace = decoded_active[idx, 0, :]
        axes[0, i].plot(trace, 'r-', linewidth=2)
        axes[0, i].set_title(f'Active #{idx}', fontsize=11, fontweight='bold')
        axes[0, i].grid(True, alpha=0.3)
        if i == 0:
            axes[0, i].set_ylabel('Activity', fontsize=11)
    
    # Inactive examples
    for i, idx in enumerate(inactive_indices):
        trace = decoded_inactive[idx, 0, :]
        axes[1, i].plot(trace, 'b-', linewidth=2)
        axes[1, i].set_title(f'Inactive #{idx}', fontsize=11, fontweight='bold')
        axes[1, i].set_xlabel('Time', fontsize=10)
        axes[1, i].grid(True, alpha=0.3)
        if i == 0:
            axes[1, i].set_ylabel('Activity', fontsize=11)
    
    plt.tight_layout()
    return fig


def main():
    print("="*70)
    print("🎯 DUAL CODEBOOK ANALYSIS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Initialize W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={**MODEL_CONFIG, **UMAP_PARAMS},
        tags=["dual-codebook", "analysis", "umap"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # 1. Carica modello
        model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # 2. Analizza utilizzo del codebook su un dataset
        print("\n" + "="*70)
        print("📊 ANALISI UTILIZZO CODEBOOK")
        print("="*70)
        
        # Crea un dataloader per analizzare l'utilizzo
        try:
            train_loader, _, _ = create_simple_calcium_dataloaders(
                batch_size=32,
                test_split=0.2,
                num_workers=0,
                window_size=60,
                stride=50,
                min_neurons=MODEL_CONFIG['num_neurons'],
                use_multi_session=False,
                use_neuron_sliding=False
            )
            
            usage_info = analyze_codebook_usage(model, train_loader, device, max_samples=1000)
            
            # Log statistiche utilizzo
            wandb.log({
                "usage/active_used": usage_info['usage_stats']['active_used'],
                "usage/active_total": usage_info['usage_stats']['active_total'],
                "usage/active_pct": usage_info['usage_stats']['active_usage_pct'],
                "usage/inactive_used": usage_info['usage_stats']['inactive_used'],
                "usage/inactive_total": usage_info['usage_stats']['inactive_total'],
                "usage/inactive_pct": usage_info['usage_stats']['inactive_usage_pct'],
                "usage/total_used": usage_info['usage_stats']['total_used'],
                "usage/total_total": usage_info['usage_stats']['total_total'],
                "usage/total_pct": usage_info['usage_stats']['total_usage_pct'],
            })
            
        except Exception as e:
            print(f"⚠️  Impossibile analizzare utilizzo: {e}")
            print("   Procedendo con tutti gli embeddings...")
            usage_info = None
        
        # 3. Estrai embeddings (filtrati se usage_info disponibile)
        emb_active, emb_inactive, indices_info = extract_separate_embeddings(model, usage_info)
        
        # 4. Decodifica codebook (solo utilizzati se disponibile)
        used_active_idx = usage_info['used_active_indices'] if usage_info else None
        used_inactive_idx = usage_info['used_inactive_indices'] if usage_info else None
        
        decoded_active = decode_codebook(model, device, 'active', temporal_length=15, used_indices=used_active_idx)
        decoded_inactive = decode_codebook(model, device, 'inactive', temporal_length=15, used_indices=used_inactive_idx)
        
        # 5. Analizza caratteristiche
        print("\n📊 Analizzando caratteristiche...")
        chars_active = analyze_characteristics(decoded_active)
        chars_inactive = analyze_characteristics(decoded_inactive)
        
        # 6. Comparison plot
        print("\n🎨 Creando comparison plot...")
        fig_comparison = plot_dual_codebook_comparison(
            emb_active, emb_inactive, chars_active, chars_inactive
        )
        wandb.log({"comparison/characteristics": wandb.Image(fig_comparison)})
        plt.close(fig_comparison)
        
        # 7. UMAP analysis
        if UMAP_AVAILABLE:
            print("\n🔬 UMAP analysis...")
            
            if len(emb_active) > 0:
                umap_active_2d = perform_umap_on_codebook(emb_active, n_components=2)
            else:
                umap_active_2d = None
                
            if len(emb_inactive) > 0:
                umap_inactive_2d = perform_umap_on_codebook(emb_inactive, n_components=2)
            else:
                umap_inactive_2d = None
            
            if umap_active_2d is not None or umap_inactive_2d is not None:
                fig_umap = plot_umap_dual_comparison(
                    umap_active_2d, umap_inactive_2d, chars_active, chars_inactive
                )
                wandb.log({"umap/dual_comparison_2d": wandb.Image(fig_umap)})
                plt.close(fig_umap)
        
        # 8. Hierarchical clustering (solo embeddings utilizzati)
        print("\n🌳 Hierarchical clustering...")
        fig_dendro = plot_hierarchical_clustering_dual(emb_active, emb_inactive, usage_info)
        wandb.log({"clustering/dendrograms": wandb.Image(fig_dendro)})
        plt.close(fig_dendro)
        
        # 9. Example reconstructions
        print("\n🎨 Plotting example reconstructions...")
        if len(decoded_active) > 0 and len(decoded_inactive) > 0:
            fig_examples = plot_example_reconstructions(decoded_active, decoded_inactive, n_examples=5)
            wandb.log({"examples/reconstructions": wandb.Image(fig_examples)})
            plt.close(fig_examples)
        
        # 10. Statistics
        print("\n📊 Logging statistics...")
        
        stats_dict = {}
        if len(chars_active['amplitude']) > 0:
            stats_dict.update({
                "stats/active_mean_amplitude": chars_active['amplitude'].mean(),
                "stats/active_mean_variance": chars_active['variance'].mean(),
                "stats/active_mean_energy": chars_active['energy'].mean(),
            })
        if len(chars_inactive['amplitude']) > 0:
            stats_dict.update({
                "stats/inactive_mean_amplitude": chars_inactive['amplitude'].mean(),
                "stats/inactive_mean_variance": chars_inactive['variance'].mean(),
                "stats/inactive_mean_energy": chars_inactive['energy'].mean(),
            })
        
        if stats_dict:
            wandb.log(stats_dict)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETATA!")
        print(f"   W&B: {wandb.run.url}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()