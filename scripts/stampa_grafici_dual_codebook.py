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
    python scripts/analyze_dual_codebook.py
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

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/dual_codebook/best_dual_model.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-dual-analysis"
WANDB_RUN_NAME = f"dual-analysis-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

MODEL_CONFIG = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 32,
    'embedding_dim': 32,
    'commitment_cost': 0.25,
    'dropout_rate': 0.1,
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


def extract_separate_embeddings(model):
    """Estrae embeddings dei due codebook separatamente"""
    vq = model.vector_quantization
    
    embeddings_active = vq.embedding_active.weight.data.cpu().numpy()
    embeddings_inactive = vq.embedding_inactive.weight.data.cpu().numpy()
    
    print(f"✅ Embeddings estratti:")
    print(f"   Active codebook: {embeddings_active.shape}")
    print(f"   Inactive codebook: {embeddings_inactive.shape}")
    
    return embeddings_active, embeddings_inactive


def decode_codebook(model, device, codebook_type='active', temporal_length=15):
    """
    Decodifica un intero codebook
    
    Args:
        codebook_type: 'active' o 'inactive'
    """
    vq = model.vector_quantization
    
    if codebook_type == 'active':
        embeddings = vq.embedding_active.weight.data
        n_codes = vq.n_e_per_codebook
    else:
        embeddings = vq.embedding_inactive.weight.data
        n_codes = vq.n_e_per_codebook
    
    print(f"\n🔍 Decodificando codebook {codebook_type.upper()} ({n_codes} codici)...")
    
    decoded = []
    
    model.eval()
    with torch.no_grad():
        for i in range(n_codes):
            code_embedding = embeddings[i]
            sequence = code_embedding.unsqueeze(1).repeat(1, temporal_length)
            sequence = sequence.unsqueeze(0).to(device)
            
            reconstruction = model.decode(sequence)
            decoded.append(reconstruction.cpu().numpy()[0])
            
            if (i + 1) % 50 == 0:
                print(f"   Processati {i+1}/{n_codes}...")
    
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
    UMAP comparison tra i due codebook
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('🎯 UMAP: Active vs Inactive Codebook', fontsize=16, fontweight='bold')
    
    # 1. Active codebook
    scatter1 = axes[0].scatter(umap_active[:, 0], umap_active[:, 1],
                              c=chars_active['amplitude'], cmap='Reds',
                              s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
    axes[0].set_title('Active Codebook', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    plt.colorbar(scatter1, ax=axes[0], label='Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Inactive codebook
    scatter2 = axes[1].scatter(umap_inactive[:, 0], umap_inactive[:, 1],
                              c=chars_inactive['amplitude'], cmap='Blues',
                              s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
    axes[1].set_title('Inactive Codebook', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    plt.colorbar(scatter2, ax=axes[1], label='Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Combined (different colors for active/inactive)
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


def plot_hierarchical_clustering_dual(emb_active, emb_inactive):
    """
    Dendrogramma gerarchico combinato per i due codebook
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    fig.suptitle('🌳 Hierarchical Clustering: Combined Dual Codebook', 
                 fontsize=16, fontweight='bold')
    
    # Combina i due codebook con etichette
    combined_embeddings = np.vstack([emb_active, emb_inactive])
    n_active = len(emb_active)
    n_inactive = len(emb_inactive)
    
    # Calcola distanze e linkage sul dataset combinato
    distances_combined = pdist(combined_embeddings, metric='cosine')
    linkage_combined = linkage(distances_combined, method='ward')
    
    # Crea il dendrogramma
    dendro = dendrogram(
        linkage_combined, 
        ax=ax, 
        color_threshold=0.5*max(linkage_combined[:,2]),
        no_labels=True  # Disabilita le etichette numeriche di default
    )
    
    ax.set_title('Combined Codebook Dendrogram (Red: Active, Blue: Inactive)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Code Index (0-511: Active | 512-1023: Inactive)', fontsize=11)
    ax.set_ylabel('Distance', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Aggiungi linee verticali per separare visivamente i due codebook
    ax.axvline(x=n_active * 10, color='gray', linestyle='--', 
               linewidth=2, alpha=0.5, label='Active/Inactive boundary')
    ax.legend()
    
    # Aggiungi annotazioni
    ax.text(0.25, 0.95, 'ACTIVE CODEBOOK', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            color='red', ha='center')
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
        
        # 2. Estrai embeddings
        emb_active, emb_inactive = extract_separate_embeddings(model)
        
        # 3. Decodifica codebook
        decoded_active = decode_codebook(model, device, 'active', temporal_length=15)
        decoded_inactive = decode_codebook(model, device, 'inactive', temporal_length=15)
        
        # 4. Analizza caratteristiche
        print("\n📊 Analizzando caratteristiche...")
        chars_active = analyze_characteristics(decoded_active)
        chars_inactive = analyze_characteristics(decoded_inactive)
        
        # 5. Comparison plot
        print("\n🎨 Creando comparison plot...")
        fig_comparison = plot_dual_codebook_comparison(
            emb_active, emb_inactive, chars_active, chars_inactive
        )
        wandb.log({"comparison/characteristics": wandb.Image(fig_comparison)})
        plt.close(fig_comparison)
        
        # 6. UMAP analysis
        if UMAP_AVAILABLE:
            print("\n🔬 UMAP analysis...")
            
            umap_active_2d = perform_umap_on_codebook(emb_active, n_components=2)
            umap_inactive_2d = perform_umap_on_codebook(emb_inactive, n_components=2)
            
            fig_umap = plot_umap_dual_comparison(
                umap_active_2d, umap_inactive_2d, chars_active, chars_inactive
            )
            wandb.log({"umap/dual_comparison_2d": wandb.Image(fig_umap)})
            plt.close(fig_umap)
        
        # 7. Hierarchical clustering
        print("\n🌳 Hierarchical clustering...")
        fig_dendro = plot_hierarchical_clustering_dual(emb_active, emb_inactive)
        wandb.log({"clustering/dendrograms": wandb.Image(fig_dendro)})
        plt.close(fig_dendro)
        
        # 8. Example reconstructions
        print("\n🎨 Plotting example reconstructions...")
        fig_examples = plot_example_reconstructions(decoded_active, decoded_inactive, n_examples=5)
        wandb.log({"examples/reconstructions": wandb.Image(fig_examples)})
        plt.close(fig_examples)
        
        # 9. Statistics
        print("\n📊 Logging statistics...")
        
        wandb.log({
            "stats/active_mean_amplitude": chars_active['amplitude'].mean(),
            "stats/inactive_mean_amplitude": chars_inactive['amplitude'].mean(),
            "stats/active_mean_variance": chars_active['variance'].mean(),
            "stats/inactive_mean_variance": chars_inactive['variance'].mean(),
            "stats/active_mean_energy": chars_active['energy'].mean(),
            "stats/inactive_mean_energy": chars_inactive['energy'].mean(),
        })
        
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