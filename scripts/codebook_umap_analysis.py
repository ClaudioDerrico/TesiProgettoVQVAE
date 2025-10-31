#!/usr/bin/env python3
"""
UMAP Analysis del Codebook VQ-VAE

Analizza la struttura del codebook usando UMAP (Uniform Manifold Approximation and Projection):
1. UMAP visualization (2D e 3D)
2. Confronto UMAP vs PCA vs t-SNE
3. Multiple metriche di distanza (cosine, euclidean)
4. Analisi dei vicini (nearest neighbors)
5. Clustering con UMAP
6. Decodifica dei codici per capire cosa rappresentano

Vantaggi di UMAP:
- Preserva struttura globale E locale
- Più veloce di t-SNE
- Deterministico (con seed fisso)
- Migliore per clustering
- Distanze più significative

Uso:
    python scripts/codebook_umap_analysis.py
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D

# ✅ UMAP import
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("⚠️  UMAP not installed. Install with: pip install umap-learn")
    UMAP_AVAILABLE = False
    sys.exit(1)

from models.vqvae import CalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/best_single_neuron_model_32.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-codebook-umap-analysis"
WANDB_RUN_NAME = f"umap-analysis-32codes"

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
}

# UMAP parameters
UMAP_N_NEIGHBORS = 5      # ✅ Per 32 codici, usa valori bassi (5-10)
UMAP_MIN_DIST = 0.1       # Distanza minima tra punti
UMAP_METRIC = 'cosine'    # 'cosine', 'euclidean', 'manhattan'
UMAP_RANDOM_STATE = 42

# ============================================================================
# FUNZIONI
# ============================================================================

def load_model(checkpoint_path, config, device):
    """Carica il modello allenato"""
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
    else:
        model.load_state_dict(checkpoint)
        print("✅ Modello caricato")
    
    model = model.to(device)
    model.eval()
    
    return model


def extract_codebook_embeddings(model):
    """Estrae gli embeddings del codebook"""
    embeddings = model.vector_quantization.embedding.weight.data.cpu().numpy()
    print(f"✅ Codebook embeddings estratti: {embeddings.shape}")
    return embeddings


def decode_all_codes(model, device, temporal_length=15):
    """
    Decodifica TUTTI i codici del codebook per capire cosa rappresentano
    """
    num_codes = model.vector_quantization.n_e
    embedding_dim = model.vector_quantization.e_dim
    
    print(f"\n🔍 Decodificando tutti i {num_codes} codici...")
    
    decoded_codes = []
    
    model.eval()
    with torch.no_grad():
        for code_idx in range(num_codes):
            code_embedding = model.vector_quantization.embedding.weight[code_idx]
            sequence = code_embedding.unsqueeze(1).repeat(1, temporal_length)
            sequence = sequence.unsqueeze(0).to(device)
            
            reconstruction = model.decode(sequence)
            decoded_codes.append(reconstruction.cpu().numpy()[0])
            
            if (code_idx + 1) % 10 == 0:
                print(f"   Processati {code_idx + 1}/{num_codes} codici...")
    
    decoded_codes = np.array(decoded_codes)
    print(f"✅ Decodifiche completate: {decoded_codes.shape}")
    
    return decoded_codes


def analyze_code_characteristics(decoded_codes):
    """Analizza le caratteristiche di ogni codice"""
    num_codes = decoded_codes.shape[0]
    
    characteristics = {
        'amplitude': [],
        'mean_activity': [],
        'variance': [],
        'max_value': [],
        'min_value': [],
        'num_peaks': [],
        'energy': []
    }
    
    for code_idx in range(num_codes):
        trace = decoded_codes[code_idx, 0, :]
        
        amplitude = np.max(trace) - np.min(trace)
        characteristics['amplitude'].append(amplitude)
        
        mean_activity = np.mean(trace)
        characteristics['mean_activity'].append(mean_activity)
        
        variance = np.var(trace)
        characteristics['variance'].append(variance)
        
        characteristics['max_value'].append(np.max(trace))
        characteristics['min_value'].append(np.min(trace))
        
        threshold = mean_activity + 2 * np.std(trace)
        peak_mask = trace > threshold
        num_peaks = np.sum(np.diff(peak_mask.astype(int)) == 1)
        characteristics['num_peaks'].append(num_peaks)
        
        energy = np.sum(trace ** 2)
        characteristics['energy'].append(energy)
    
    for key in characteristics:
        characteristics[key] = np.array(characteristics[key])
    
    print(f"✅ Caratteristiche calcolate per {num_codes} codici")
    
    return characteristics


def perform_umap(embeddings, n_neighbors=15, min_dist=0.1, 
                 metric='cosine', n_components=2, random_state=42):
    """
    Esegue UMAP sugli embeddings
    
    Args:
        embeddings: (num_codes, embedding_dim)
        n_neighbors: numero di vicini (5-15 per piccoli dataset)
        min_dist: distanza minima tra punti (0.1 default)
        metric: 'cosine', 'euclidean', 'manhattan'
        n_components: 2D o 3D
        random_state: seed per riproducibilità
    
    Returns:
        umap_coords: transformed coordinates
        umap_model: fitted UMAP model
    """
    print(f"\n🔬 Eseguendo UMAP...")
    print(f"   Input shape: {embeddings.shape}")
    print(f"   n_neighbors: {n_neighbors}")
    print(f"   min_dist: {min_dist}")
    print(f"   metric: {metric}")
    print(f"   n_components: {n_components}")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True
    )
    
    umap_coords = reducer.fit_transform(embeddings)
    
    print(f"✅ UMAP completato: {umap_coords.shape}")
    
    return umap_coords, reducer


def compare_dimensionality_reduction(embeddings):
    """
    Confronta UMAP vs PCA vs t-SNE
    
    Returns:
        dict con coordinate di tutti i metodi
    """
    print(f"\n🔬 Confrontando metodi di dimensionality reduction...")
    
    results = {}
    
    # 1. UMAP
    print("\n   🔵 UMAP...")
    umap_coords, _ = perform_umap(embeddings, 
                                   n_neighbors=UMAP_N_NEIGHBORS,
                                   min_dist=UMAP_MIN_DIST,
                                   metric=UMAP_METRIC,
                                   n_components=2,
                                   random_state=UMAP_RANDOM_STATE)
    results['umap'] = umap_coords
    
    # 2. PCA
    print("\n   🟢 PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(embeddings)
    results['pca'] = pca_coords
    print(f"      Variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # 3. t-SNE
    print("\n   🟠 t-SNE...")
    perplexity = min(5, embeddings.shape[0] - 1)  # Perplexity adattativa
    tsne = TSNE(n_components=2, perplexity=perplexity, 
                n_iter=3000, random_state=42, verbose=0)
    tsne_coords = tsne.fit_transform(embeddings)
    results['tsne'] = tsne_coords
    
    print(f"✅ Tutti i metodi completati!")
    
    return results


def plot_umap_2d_visualization(umap_coords, characteristics):
    """
    Visualizzazioni UMAP 2D con diverse colorazioni
    """
    num_codes = umap_coords.shape[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🎯 UMAP 2D Visualization del Codebook', 
                 fontsize=16, fontweight='bold')
    
    # 1. Colored by code index
    scatter1 = axes[0, 0].scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                  c=range(num_codes), cmap='tab20', s=100, alpha=0.7,
                                  edgecolors='black', linewidths=0.5)
    for i in range(num_codes):
        axes[0, 0].annotate(str(i), (umap_coords[i, 0], umap_coords[i, 1]),
                           fontsize=8, ha='center', va='center', fontweight='bold')
    axes[0, 0].set_title('Colored by Code Index', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('UMAP 1')
    axes[0, 0].set_ylabel('UMAP 2')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Code Index')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Colored by amplitude
    scatter2 = axes[0, 1].scatter(umap_coords[:, 0], umap_coords[:, 1],
                                  c=characteristics['amplitude'], cmap='viridis', 
                                  s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
    axes[0, 1].set_title('Colored by Amplitude', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('UMAP 1')
    axes[0, 1].set_ylabel('UMAP 2')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Colored by mean activity
    scatter3 = axes[0, 2].scatter(umap_coords[:, 0], umap_coords[:, 1],
                                  c=characteristics['mean_activity'], cmap='coolwarm', 
                                  s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
    axes[0, 2].set_title('Colored by Mean Activity', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('UMAP 1')
    axes[0, 2].set_ylabel('UMAP 2')
    plt.colorbar(scatter3, ax=axes[0, 2], label='Mean Activity')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Colored by variance
    scatter4 = axes[1, 0].scatter(umap_coords[:, 0], umap_coords[:, 1],
                                  c=characteristics['variance'], cmap='plasma', 
                                  s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
    axes[1, 0].set_title('Colored by Variance', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    plt.colorbar(scatter4, ax=axes[1, 0], label='Variance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Colored by num_peaks
    scatter5 = axes[1, 1].scatter(umap_coords[:, 0], umap_coords[:, 1],
                                  c=characteristics['num_peaks'], cmap='Spectral', 
                                  s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
    axes[1, 1].set_title('Colored by Number of Peaks', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('UMAP 1')
    axes[1, 1].set_ylabel('UMAP 2')
    plt.colorbar(scatter5, ax=axes[1, 1], label='Num Peaks')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Colored by energy
    scatter6 = axes[1, 2].scatter(umap_coords[:, 0], umap_coords[:, 1],
                                  c=characteristics['energy'], cmap='magma', 
                                  s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
    axes[1, 2].set_title('Colored by Energy', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('UMAP 1')
    axes[1, 2].set_ylabel('UMAP 2')
    plt.colorbar(scatter6, ax=axes[1, 2], label='Energy')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_umap_3d_visualization(umap_coords, characteristics):
    """
    Visualizzazioni UMAP 3D con diverse colorazioni
    """
    num_codes = umap_coords.shape[0]
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('🎯 UMAP 3D Visualization del Codebook', 
                 fontsize=16, fontweight='bold')
    
    # 1. Colored by code index
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter1 = ax1.scatter(umap_coords[:, 0], umap_coords[:, 1], umap_coords[:, 2],
                          c=range(num_codes), cmap='tab20', s=50, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
    ax1.set_title('Colored by Code Index', fontweight='bold')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_zlabel('UMAP 3')
    plt.colorbar(scatter1, ax=ax1, label='Code Index', shrink=0.5)
    
    # 2. Colored by amplitude
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    scatter2 = ax2.scatter(umap_coords[:, 0], umap_coords[:, 1], umap_coords[:, 2],
                          c=characteristics['amplitude'], cmap='viridis', s=50, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
    ax2.set_title('Colored by Amplitude', fontweight='bold')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.set_zlabel('UMAP 3')
    plt.colorbar(scatter2, ax=ax2, label='Amplitude', shrink=0.5)
    
    # 3. Colored by mean activity
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    scatter3 = ax3.scatter(umap_coords[:, 0], umap_coords[:, 1], umap_coords[:, 2],
                          c=characteristics['mean_activity'], cmap='coolwarm', s=50, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
    ax3.set_title('Colored by Mean Activity', fontweight='bold')
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    ax3.set_zlabel('UMAP 3')
    plt.colorbar(scatter3, ax=ax3, label='Mean Activity', shrink=0.5)
    
    # 4. Colored by variance
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    scatter4 = ax4.scatter(umap_coords[:, 0], umap_coords[:, 1], umap_coords[:, 2],
                          c=characteristics['variance'], cmap='plasma', s=50, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
    ax4.set_title('Colored by Variance', fontweight='bold')
    ax4.set_xlabel('UMAP 1')
    ax4.set_ylabel('UMAP 2')
    ax4.set_zlabel('UMAP 3')
    plt.colorbar(scatter4, ax=ax4, label='Variance', shrink=0.5)
    
    # 5. Colored by num_peaks
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    scatter5 = ax5.scatter(umap_coords[:, 0], umap_coords[:, 1], umap_coords[:, 2],
                          c=characteristics['num_peaks'], cmap='Spectral', s=50, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
    ax5.set_title('Colored by Number of Peaks', fontweight='bold')
    ax5.set_xlabel('UMAP 1')
    ax5.set_ylabel('UMAP 2')
    ax5.set_zlabel('UMAP 3')
    plt.colorbar(scatter5, ax=ax5, label='Num Peaks', shrink=0.5)
    
    # 6. Colored by energy
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    scatter6 = ax6.scatter(umap_coords[:, 0], umap_coords[:, 1], umap_coords[:, 2],
                          c=characteristics['energy'], cmap='magma', s=50, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
    ax6.set_title('Colored by Energy', fontweight='bold')
    ax6.set_xlabel('UMAP 1')
    ax6.set_ylabel('UMAP 2')
    ax6.set_zlabel('UMAP 3')
    plt.colorbar(scatter6, ax=ax6, label='Energy', shrink=0.5)
    
    plt.tight_layout()
    
    return fig


def plot_method_comparison(all_coords, characteristics):
    """
    Confronto visivo: UMAP vs PCA vs t-SNE
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('🔬 Comparison: UMAP vs PCA vs t-SNE', 
                 fontsize=16, fontweight='bold')
    
    methods = ['umap', 'pca', 'tsne']
    titles = ['UMAP', 'PCA', 't-SNE']
    
    # Usa amplitude per colorare
    colors = characteristics['amplitude']
    
    for idx, (method, title) in enumerate(zip(methods, titles)):
        coords = all_coords[method]
        
        scatter = axes[idx].scatter(coords[:, 0], coords[:, 1],
                                   c=colors, cmap='viridis', s=100, alpha=0.7,
                                   edgecolors='black', linewidths=0.5)
        
        # Aggiungi labels
        for i in range(len(coords)):
            axes[idx].annotate(str(i), (coords[i, 0], coords[i, 1]),
                             fontsize=7, ha='center', va='center', fontweight='bold')
        
        axes[idx].set_title(title, fontsize=13, fontweight='bold')
        axes[idx].set_xlabel(f'{title} 1')
        axes[idx].set_ylabel(f'{title} 2')
        axes[idx].grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=axes[idx], label='Amplitude')
    
    plt.tight_layout()
    
    return fig


def analyze_umap_clusters(umap_coords, characteristics, decoded_codes, 
                          distance_threshold=0.5):
    """
    Identifica cluster nello spazio UMAP e analizza le loro proprietà
    
    Args:
        umap_coords: coordinate UMAP
        characteristics: caratteristiche dei codici
        decoded_codes: tracce decodificate
        distance_threshold: threshold per clustering
    """
    from sklearn.cluster import DBSCAN
    from scipy.spatial.distance import cdist
    
    print(f"\n🔍 Analizzando cluster UMAP...")
    
    # Clustering DBSCAN su coordinate UMAP
    clustering = DBSCAN(eps=distance_threshold, min_samples=2)
    labels = clustering.fit_predict(umap_coords)
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"✅ Trovati {n_clusters} cluster, {n_noise} outliers")
    
    # Analizza ogni cluster
    cluster_info = {}
    
    for cluster_id in unique_labels:
        if cluster_id == -1:
            # Outliers
            continue
        
        # Codici in questo cluster
        cluster_mask = labels == cluster_id
        cluster_codes = np.where(cluster_mask)[0]
        
        print(f"\n   Cluster {cluster_id}:")
        print(f"      Codes: {cluster_codes.tolist()}")
        
        # Caratteristiche medie del cluster
        cluster_chars = {}
        for key in characteristics:
            values = characteristics[key][cluster_mask]
            cluster_chars[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Esempio: amplitude
        amp_mean = cluster_chars['amplitude']['mean']
        print(f"      Amplitude mean: {amp_mean:.3f}")
        
        cluster_info[cluster_id] = {
            'codes': cluster_codes.tolist(),
            'size': len(cluster_codes),
            'characteristics': cluster_chars
        }
    
    # Visualizza cluster
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot con colori per cluster
    for cluster_id in unique_labels:
        if cluster_id == -1:
            # Outliers in nero
            cluster_mask = labels == cluster_id
            ax.scatter(umap_coords[cluster_mask, 0], umap_coords[cluster_mask, 1],
                      c='black', s=100, alpha=0.5, marker='x', label='Outliers')
        else:
            cluster_mask = labels == cluster_id
            ax.scatter(umap_coords[cluster_mask, 0], umap_coords[cluster_mask, 1],
                      s=150, alpha=0.7, edgecolors='black', linewidths=1,
                      label=f'Cluster {cluster_id}')
    
    # Labels
    for i in range(len(umap_coords)):
        ax.annotate(str(i), (umap_coords[i, 0], umap_coords[i, 1]),
                   fontsize=9, ha='center', va='center', fontweight='bold')
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'UMAP Clustering (DBSCAN, eps={distance_threshold})\n'
                 f'{n_clusters} clusters, {n_noise} outliers',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, cluster_info, labels


def plot_umap_with_connections(umap_coords, embeddings, k=3):
    """
    Visualizza UMAP con connessioni tra nearest neighbors
    """
    from sklearn.neighbors import NearestNeighbors
    
    print(f"\n🔗 Creando visualizzazione con k={k} nearest neighbors...")
    
    # Trova k-NN nello spazio ORIGINALE degli embeddings
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Disegna connessioni
    for i in range(len(umap_coords)):
        for j in range(1, k+1):  # Skip primo (se stesso)
            neighbor_idx = indices[i, j]
            
            # Linea da i a neighbor
            ax.plot([umap_coords[i, 0], umap_coords[neighbor_idx, 0]],
                   [umap_coords[i, 1], umap_coords[neighbor_idx, 1]],
                   'gray', alpha=0.3, linewidth=0.5, zorder=1)
    
    # Disegna punti sopra
    scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                        c=range(len(umap_coords)), cmap='tab20',
                        s=150, alpha=0.8, edgecolors='black', linewidths=1, zorder=2)
    
    # Labels
    for i in range(len(umap_coords)):
        ax.annotate(str(i), (umap_coords[i, 0], umap_coords[i, 1]),
                   fontsize=9, ha='center', va='center', fontweight='bold', zorder=3)
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'UMAP with {k}-Nearest Neighbor Connections\n(based on original embedding space)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Code Index')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def main():
    print("="*70)
    print("🎯 UMAP ANALYSIS DEL CODEBOOK VQ-VAE")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Embedding dim: {MODEL_CONFIG['embedding_dim']}")
    print(f"   UMAP n_neighbors: {UMAP_N_NEIGHBORS}")
    print(f"   UMAP min_dist: {UMAP_MIN_DIST}")
    print(f"   UMAP metric: {UMAP_METRIC}")
    print(f"   W&B Project: {WANDB_PROJECT}")
    
    if not UMAP_AVAILABLE:
        print("\n❌ UMAP non disponibile. Installa con:")
        print("   pip install umap-learn")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Inizializza W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "checkpoint_path": CHECKPOINT_PATH,
            "umap_n_neighbors": UMAP_N_NEIGHBORS,
            "umap_min_dist": UMAP_MIN_DIST,
            "umap_metric": UMAP_METRIC,
            **MODEL_CONFIG
        },
        tags=["umap", "codebook-analysis", "similarity", "clustering"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # 1. Carica modello
        model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # 2. Estrai embeddings del codebook
        embeddings = extract_codebook_embeddings(model)
        
        # 3. Decodifica tutti i codici
        decoded_codes = decode_all_codes(model, device, temporal_length=15)
        
        # 4. Analizza caratteristiche dei codici
        characteristics = analyze_code_characteristics(decoded_codes)
        
        # 5. UMAP 2D
        print("\n" + "="*70)
        print("🎯 UMAP 2D")
        print("="*70)
        umap_coords_2d, umap_model_2d = perform_umap(
            embeddings, 
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            n_components=2,
            random_state=UMAP_RANDOM_STATE
        )
        
        # Visualizza UMAP 2D
        fig_umap_2d = plot_umap_2d_visualization(umap_coords_2d, characteristics)
        wandb.log({"umap/2d_visualization": wandb.Image(fig_umap_2d)})
        plt.close(fig_umap_2d)
        
        # 6. UMAP 3D
        print("\n" + "="*70)
        print("🎯 UMAP 3D")
        print("="*70)
        umap_coords_3d, umap_model_3d = perform_umap(
            embeddings, 
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            n_components=3,
            random_state=UMAP_RANDOM_STATE
        )
        
        # Visualizza UMAP 3D
        fig_umap_3d = plot_umap_3d_visualization(umap_coords_3d, characteristics)
        wandb.log({"umap/3d_visualization": wandb.Image(fig_umap_3d)})
        plt.close(fig_umap_3d)
        
        # 7. Confronto UMAP vs PCA vs t-SNE
        print("\n" + "="*70)
        print("🔬 CONFRONTO: UMAP vs PCA vs t-SNE")
        print("="*70)
        all_coords = compare_dimensionality_reduction(embeddings)
        
        fig_comparison = plot_method_comparison(all_coords, characteristics)
        wandb.log({"comparison/umap_vs_pca_vs_tsne": wandb.Image(fig_comparison)})
        plt.close(fig_comparison)
        
        # 8. Clustering su UMAP
        print("\n" + "="*70)
        print("🔍 CLUSTERING SU SPAZIO UMAP")
        print("="*70)
        fig_clusters, cluster_info, labels = analyze_umap_clusters(
            umap_coords_2d, characteristics, decoded_codes, 
            distance_threshold=0.5
        )
        wandb.log({"umap/clustering": wandb.Image(fig_clusters)})
        plt.close(fig_clusters)
        
        # Log cluster info
        for cluster_id, info in cluster_info.items():
            wandb.log({
                f"clusters/cluster_{cluster_id}_size": info['size'],
                f"clusters/cluster_{cluster_id}_codes": str(info['codes'])
            })
        
        # 9. UMAP con connessioni k-NN
        print("\n" + "="*70)
        print("🔗 UMAP CON NEAREST NEIGHBOR CONNECTIONS")
        print("="*70)
        fig_knn = plot_umap_with_connections(umap_coords_2d, embeddings, k=3)
        wandb.log({"umap/knn_connections": wandb.Image(fig_knn)})
        plt.close(fig_knn)
        
        # 10. Metriche aggregate
        print("\n📊 Loggando metriche aggregate...")
        
        # Calcola similarità media nello spazio UMAP vs embedding originale
        from scipy.spatial.distance import pdist
        
        # Distanze nello spazio UMAP
        umap_distances = pdist(umap_coords_2d, metric='euclidean')
        
        # Distanze nello spazio originale
        original_distances = pdist(embeddings, metric='cosine')
        
        # Correlazione tra distanze (trustworthiness)
        from scipy.stats import spearmanr
        corr, _ = spearmanr(umap_distances, original_distances)
        
        wandb.log({
            "umap/distance_correlation": corr,
            "umap/mean_distance_umap": np.mean(umap_distances),
            "umap/std_distance_umap": np.std(umap_distances),
            "umap/mean_distance_original": np.mean(original_distances),
            "umap/n_clusters": len(cluster_info),
        })
        
        print(f"\n📊 UMAP Statistics:")
        print(f"   Distance correlation (Spearman): {corr:.3f}")
        print(f"   Mean distance (UMAP space): {np.mean(umap_distances):.3f}")
        print(f"   Mean distance (original space): {np.mean(original_distances):.3f}")
        print(f"   Clusters found: {len(cluster_info)}")
        
        print("\n" + "="*70)
        print("✅ UMAP ANALYSIS COMPLETATA!")
        print(f"   Tutti i risultati su W&B: {wandb.run.url}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        wandb.finish()
        print("✅ W&B run completato!")


if __name__ == "__main__":
    main()