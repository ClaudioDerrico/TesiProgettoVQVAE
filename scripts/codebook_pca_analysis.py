#!/usr/bin/env python3
"""
PCA Analysis del Codebook VQ-VAE

Analizza la struttura del codebook attraverso:
1. PCA visualization dei codici (2D e 3D)
2. Varianza spiegata per ogni componente
3. Heatmap di similarità (cosine similarity)
4. Clustering gerarchico
5. Analisi dei codici vicini (perché sono simili?)
6. Decodifica dei codici per capire cosa rappresentano
7. Loading dei componenti principali (quali feature influenzano ogni PC)

Vantaggi di PCA vs t-SNE:
- Deterministico (sempre stesso risultato)
- Più veloce
- Interpretabile (componenti principali hanno significato)
- Distanze preservate meglio
- Varianza spiegata

Uso:
    python scripts/codebook_pca_analysis.py
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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D

from models.vqvae import CalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/best_single_neuron_model_32.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-codebook-analysis"
WANDB_RUN_NAME = f"pca-analysis-32codes"

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

# PCA parameters
N_COMPONENTS_VIZ = 3  # Per visualization (2D e 3D)
N_COMPONENTS_FULL = None  # None = tutte le componenti

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
    
    Args:
        model: VQ-VAE model
        device: torch device
        temporal_length: lunghezza sequenza da decodificare
    
    Returns:
        decoded_codes: array (num_codes, num_neurons, output_time)
    """
    num_codes = model.vector_quantization.n_e
    embedding_dim = model.vector_quantization.e_dim
    
    print(f"\n🔍 Decodificando tutti i {num_codes} codici...")
    
    decoded_codes = []
    
    model.eval()
    with torch.no_grad():
        for code_idx in range(num_codes):
            # Crea sequenza con questo codice ripetuto
            code_embedding = model.vector_quantization.embedding.weight[code_idx]
            
            # Ripeti per temporal_length
            sequence = code_embedding.unsqueeze(1).repeat(1, temporal_length)  # (embedding_dim, temporal_length)
            sequence = sequence.unsqueeze(0).to(device)  # (1, embedding_dim, temporal_length)
            
            # Decodifica
            reconstruction = model.decode(sequence)
            decoded_codes.append(reconstruction.cpu().numpy()[0])  # (num_neurons, output_time)
            
            if (code_idx + 1) % 10 == 0:
                print(f"   Processati {code_idx + 1}/{num_codes} codici...")
    
    decoded_codes = np.array(decoded_codes)  # (num_codes, num_neurons, output_time)
    print(f"✅ Decodifiche completate: {decoded_codes.shape}")
    
    return decoded_codes


def compute_decoded_similarities(decoded_codes):
    """
    Calcola similarità tra codici basata sulle DECODIFICHE (non sugli embeddings)
    
    Args:
        decoded_codes: (num_codes, num_neurons, output_time)
    
    Returns:
        similarity_matrix: (num_codes, num_codes) matrice di similarità
    """
    num_codes = decoded_codes.shape[0]
    
    # Flatten le decodifiche per ogni codice
    decoded_flat = decoded_codes.reshape(num_codes, -1)  # (num_codes, neurons*time)
    
    # Normalizza
    decoded_flat_norm = decoded_flat / (np.linalg.norm(decoded_flat, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity
    similarity = np.dot(decoded_flat_norm, decoded_flat_norm.T)
    
    print(f"✅ Similarità decodifiche calcolata: {similarity.shape}")
    
    return similarity


def analyze_code_characteristics(decoded_codes):
    """
    Analizza le caratteristiche di ogni codice (ampiezza, varianza, picchi, ecc.)
    
    Args:
        decoded_codes: (num_codes, num_neurons, output_time)
    
    Returns:
        characteristics: dict con caratteristiche di ogni codice
    """
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
        trace = decoded_codes[code_idx, 0, :]  # Singolo neurone
        
        # Ampiezza
        amplitude = np.max(trace) - np.min(trace)
        characteristics['amplitude'].append(amplitude)
        
        # Mean activity
        mean_activity = np.mean(trace)
        characteristics['mean_activity'].append(mean_activity)
        
        # Variance
        variance = np.var(trace)
        characteristics['variance'].append(variance)
        
        # Max/Min
        characteristics['max_value'].append(np.max(trace))
        characteristics['min_value'].append(np.min(trace))
        
        # Numero di picchi (approssimato)
        threshold = mean_activity + 2 * np.std(trace)
        peak_mask = trace > threshold
        num_peaks = np.sum(np.diff(peak_mask.astype(int)) == 1)
        characteristics['num_peaks'].append(num_peaks)
        
        # Energy (sum of squared values)
        energy = np.sum(trace ** 2)
        characteristics['energy'].append(energy)
    
    # Converti in array
    for key in characteristics:
        characteristics[key] = np.array(characteristics[key])
    
    print(f"✅ Caratteristiche calcolate per {num_codes} codici")
    
    return characteristics


def perform_pca(embeddings, n_components=None):
    """
    Esegue PCA sugli embeddings
    
    Args:
        embeddings: (num_codes, embedding_dim)
        n_components: numero di componenti (None = tutte)
    
    Returns:
        pca_model: fitted PCA model
        pca_coords: transformed coordinates
    """
    print(f"\n🔬 Eseguendo PCA...")
    print(f"   Input shape: {embeddings.shape}")
    
    if n_components is None:
        n_components = min(embeddings.shape)
    
    print(f"   N components: {n_components}")
    
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(embeddings)
    
    print(f"✅ PCA completato: {pca_coords.shape}")
    
    # Varianza spiegata
    print(f"\n📊 Varianza spiegata:")
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    for i in range(min(10, len(pca.explained_variance_ratio_))):
        print(f"   PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} "
              f"(cumulative: {cumulative_var[i]:.4f})")
    
    return pca, pca_coords


def plot_pca_2d_visualization(pca_coords, characteristics):
    """
    Crea visualizzazioni PCA 2D con diverse colorazioni
    
    Args:
        pca_coords: (num_codes, n_components) coordinate PCA
        characteristics: dict con caratteristiche dei codici
    """
    num_codes = pca_coords.shape[0]
    
    # ========================================================================
    # FIGURA: PCA 2D con multiple colorazioni
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PCA 2D Visualization del Codebook', fontsize=16, fontweight='bold')
    
    # 1. Colored by code index
    scatter1 = axes[0, 0].scatter(pca_coords[:, 0], pca_coords[:, 1], 
                                  c=range(num_codes), cmap='tab20', s=100, alpha=0.7)
    for i in range(num_codes):
        axes[0, 0].annotate(str(i), (pca_coords[i, 0], pca_coords[i, 1]),
                           fontsize=8, ha='center', va='center')
    axes[0, 0].set_title('Colored by Code Index')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Code Index')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Colored by amplitude
    scatter2 = axes[0, 1].scatter(pca_coords[:, 0], pca_coords[:, 1],
                                  c=characteristics['amplitude'], cmap='viridis', s=100, alpha=0.7)
    axes[0, 1].set_title('Colored by Amplitude')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Colored by mean activity
    scatter3 = axes[0, 2].scatter(pca_coords[:, 0], pca_coords[:, 1],
                                  c=characteristics['mean_activity'], cmap='coolwarm', s=100, alpha=0.7)
    axes[0, 2].set_title('Colored by Mean Activity')
    axes[0, 2].set_xlabel('PC1')
    axes[0, 2].set_ylabel('PC2')
    plt.colorbar(scatter3, ax=axes[0, 2], label='Mean Activity')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Colored by variance
    scatter4 = axes[1, 0].scatter(pca_coords[:, 0], pca_coords[:, 1],
                                  c=characteristics['variance'], cmap='plasma', s=100, alpha=0.7)
    axes[1, 0].set_title('Colored by Variance')
    axes[1, 0].set_xlabel('PC1')
    axes[1, 0].set_ylabel('PC2')
    plt.colorbar(scatter4, ax=axes[1, 0], label='Variance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Colored by num_peaks
    scatter5 = axes[1, 1].scatter(pca_coords[:, 0], pca_coords[:, 1],
                                  c=characteristics['num_peaks'], cmap='Spectral', s=100, alpha=0.7)
    axes[1, 1].set_title('Colored by Number of Peaks')
    axes[1, 1].set_xlabel('PC1')
    axes[1, 1].set_ylabel('PC2')
    plt.colorbar(scatter5, ax=axes[1, 1], label='Num Peaks')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Colored by energy
    scatter6 = axes[1, 2].scatter(pca_coords[:, 0], pca_coords[:, 1],
                                  c=characteristics['energy'], cmap='magma', s=100, alpha=0.7)
    axes[1, 2].set_title('Colored by Energy')
    axes[1, 2].set_xlabel('PC1')
    axes[1, 2].set_ylabel('PC2')
    plt.colorbar(scatter6, ax=axes[1, 2], label='Energy')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_pca_3d_visualization(pca_coords, characteristics):
    """
    Crea visualizzazioni PCA 3D con diverse colorazioni
    
    Args:
        pca_coords: (num_codes, n_components) coordinate PCA
        characteristics: dict con caratteristiche dei codici
    """
    num_codes = pca_coords.shape[0]
    
    # ========================================================================
    # FIGURA: PCA 3D con multiple colorazioni
    # ========================================================================
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('PCA 3D Visualization del Codebook', fontsize=16, fontweight='bold')
    
    # 1. Colored by code index
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter1 = ax1.scatter(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
                          c=range(num_codes), cmap='tab20', s=50, alpha=0.7)
    ax1.set_title('Colored by Code Index')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    plt.colorbar(scatter1, ax=ax1, label='Code Index', shrink=0.5)
    
    # 2. Colored by amplitude
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    scatter2 = ax2.scatter(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
                          c=characteristics['amplitude'], cmap='viridis', s=50, alpha=0.7)
    ax2.set_title('Colored by Amplitude')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    plt.colorbar(scatter2, ax=ax2, label='Amplitude', shrink=0.5)
    
    # 3. Colored by mean activity
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    scatter3 = ax3.scatter(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
                          c=characteristics['mean_activity'], cmap='coolwarm', s=50, alpha=0.7)
    ax3.set_title('Colored by Mean Activity')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_zlabel('PC3')
    plt.colorbar(scatter3, ax=ax3, label='Mean Activity', shrink=0.5)
    
    # 4. Colored by variance
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    scatter4 = ax4.scatter(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
                          c=characteristics['variance'], cmap='plasma', s=50, alpha=0.7)
    ax4.set_title('Colored by Variance')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_zlabel('PC3')
    plt.colorbar(scatter4, ax=ax4, label='Variance', shrink=0.5)
    
    # 5. Colored by num_peaks
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    scatter5 = ax5.scatter(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
                          c=characteristics['num_peaks'], cmap='Spectral', s=50, alpha=0.7)
    ax5.set_title('Colored by Number of Peaks')
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.set_zlabel('PC3')
    plt.colorbar(scatter5, ax=ax5, label='Num Peaks', shrink=0.5)
    
    # 6. Colored by energy
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    scatter6 = ax6.scatter(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
                          c=characteristics['energy'], cmap='magma', s=50, alpha=0.7)
    ax6.set_title('Colored by Energy')
    ax6.set_xlabel('PC1')
    ax6.set_ylabel('PC2')
    ax6.set_zlabel('PC3')
    plt.colorbar(scatter6, ax=ax6, label='Energy', shrink=0.5)
    
    plt.tight_layout()
    
    return fig


def plot_explained_variance(pca_model):
    """
    Plotta la varianza spiegata dai componenti principali
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    explained_var = pca_model.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # 1. Varianza per componente
    axes[0].bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, color='skyblue')
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0].set_title('Variance Explained by Each PC', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Evidenzia prime 3 componenti
    for i in range(min(3, len(explained_var))):
        axes[0].bar(i + 1, explained_var[i], alpha=0.9, color='darkblue')
    
    # 2. Varianza cumulativa
    axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                marker='o', linewidth=2, markersize=6, color='darkgreen')
    axes[1].axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% threshold')
    axes[1].axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90% threshold')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_pca_loadings(pca_model, n_top_features=10):
    """
    Visualizza i loadings (contributi) delle feature originali sui primi PC
    
    Questo aiuta a capire COSA rappresentano i componenti principali
    """
    loadings = pca_model.components_  # (n_components, n_features)
    n_components = min(3, loadings.shape[0])
    
    fig, axes = plt.subplots(1, n_components, figsize=(6 * n_components, 6))
    
    if n_components == 1:
        axes = [axes]
    
    for i in range(n_components):
        # Top features per questo componente
        loading = loadings[i, :]
        top_indices = np.argsort(np.abs(loading))[-n_top_features:][::-1]
        top_loadings = loading[top_indices]
        
        colors = ['red' if x < 0 else 'green' for x in top_loadings]
        
        axes[i].barh(range(n_top_features), top_loadings, color=colors, alpha=0.7)
        axes[i].set_yticks(range(n_top_features))
        axes[i].set_yticklabels([f"Dim {idx}" for idx in top_indices])
        axes[i].set_xlabel('Loading Value', fontsize=11)
        axes[i].set_title(f'PC{i+1} Top {n_top_features} Features\n'
                         f'(Var: {pca_model.explained_variance_ratio_[i]:.3f})',
                         fontsize=12, fontweight='bold')
        axes[i].axvline(0, color='black', linewidth=0.8)
        axes[i].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    return fig


def plot_similarity_heatmap(similarity_matrix, title="Codebook Similarity Matrix"):
    """
    Crea heatmap di similarità tra codici
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap
    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    # Labels
    ax.set_xlabel('Code Index', fontsize=12)
    ax.set_ylabel('Code Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Ticks
    num_codes = similarity_matrix.shape[0]
    tick_step = max(1, num_codes // 20)  # Max 20 ticks
    ticks = np.arange(0, num_codes, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    
    plt.tight_layout()
    
    return fig


def plot_hierarchical_clustering(embeddings, similarity_matrix):
    """
    Clustering gerarchico del codebook
    """
    # Calcola linkage
    # Usa distanza = 1 - similarity
    distance_matrix = 1 - similarity_matrix
    
    # Converti in condensed distance matrix
    condensed_dist = squareform(distance_matrix, checks=False)
    
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(14, 6))
    
    dendrogram(linkage_matrix, ax=ax, 
               labels=list(range(embeddings.shape[0])),
               leaf_font_size=10)
    
    ax.set_xlabel('Code Index', fontsize=12)
    ax.set_ylabel('Distance (1 - Similarity)', fontsize=12)
    ax.set_title('Hierarchical Clustering of Codebook', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def find_nearest_neighbors(similarity_matrix, k=3):
    """
    Trova i k nearest neighbors per ogni codice
    
    Args:
        similarity_matrix: (num_codes, num_codes)
        k: numero di vicini da trovare
    
    Returns:
        neighbors: dict {code_idx: [(neighbor_idx, similarity), ...]}
    """
    num_codes = similarity_matrix.shape[0]
    neighbors = {}
    
    for code_idx in range(num_codes):
        # Ottieni similarità con tutti gli altri codici (escluso se stesso)
        similarities = similarity_matrix[code_idx].copy()
        similarities[code_idx] = -np.inf  # Escludi se stesso
        
        # Top k
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_similarities = similarities[top_k_indices]
        
        neighbors[code_idx] = list(zip(top_k_indices.tolist(), top_k_similarities.tolist()))
    
    return neighbors


def visualize_nearest_neighbors(decoded_codes, neighbors, code_idx, num_neighbors=3):
    """
    Visualizza un codice e i suoi nearest neighbors
    
    Args:
        decoded_codes: (num_codes, num_neurons, output_time)
        neighbors: dict da find_nearest_neighbors
        code_idx: indice del codice da visualizzare
        num_neighbors: numero di vicini da mostrare
    """
    fig, axes = plt.subplots(num_neighbors + 1, 1, figsize=(14, 3 * (num_neighbors + 1)))
    
    if num_neighbors == 0:
        axes = [axes]
    
    output_time = decoded_codes.shape[2]
    time_points = np.arange(output_time)
    
    # Plot del codice principale
    main_trace = decoded_codes[code_idx, 0, :]
    axes[0].plot(time_points, main_trace, 'b-', linewidth=2, label=f'Code {code_idx} (MAIN)')
    axes[0].set_ylabel('Activity')
    axes[0].set_title(f'CODE {code_idx} (Reference)', fontweight='bold', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Caratteristiche del codice principale
    main_amp = np.max(main_trace) - np.min(main_trace)
    main_mean = np.mean(main_trace)
    main_var = np.var(main_trace)
    
    stats_text = f'Amp: {main_amp:.3f}\nMean: {main_mean:.3f}\nVar: {main_var:.3f}'
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Plot dei neighbors
    neighbor_list = neighbors[code_idx][:num_neighbors]
    
    for plot_idx, (neighbor_idx, similarity) in enumerate(neighbor_list, 1):
        neighbor_trace = decoded_codes[neighbor_idx, 0, :]
        
        # Overlay con main code
        axes[plot_idx].plot(time_points, main_trace, 'b--', linewidth=1, 
                           alpha=0.5, label=f'Code {code_idx} (ref)')
        axes[plot_idx].plot(time_points, neighbor_trace, 'r-', linewidth=2,
                           label=f'Code {neighbor_idx} (neighbor)')
        
        # Calcola differenza
        diff = neighbor_trace - main_trace
        
        axes[plot_idx].fill_between(time_points, main_trace, neighbor_trace,
                                    where=(diff > 0), color='orange', alpha=0.2)
        axes[plot_idx].fill_between(time_points, main_trace, neighbor_trace,
                                    where=(diff < 0), color='purple', alpha=0.2)
        
        axes[plot_idx].set_ylabel('Activity')
        axes[plot_idx].set_title(f'Neighbor {plot_idx}: Code {neighbor_idx} (Similarity: {similarity:.3f})',
                                 fontweight='bold', fontsize=11)
        axes[plot_idx].legend(loc='upper right')
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Caratteristiche del neighbor
        neighbor_amp = np.max(neighbor_trace) - np.min(neighbor_trace)
        neighbor_mean = np.mean(neighbor_trace)
        neighbor_var = np.var(neighbor_trace)
        
        stats_text = f'Amp: {neighbor_amp:.3f}\nMean: {neighbor_mean:.3f}\nVar: {neighbor_var:.3f}'
        stats_text += f'\n\nDiff vs main:\nΔAmp: {neighbor_amp - main_amp:.3f}\n'
        stats_text += f'ΔMean: {neighbor_mean - main_mean:.3f}\nΔVar: {neighbor_var - main_var:.3f}'
        
        axes[plot_idx].text(0.02, 0.98, stats_text, transform=axes[plot_idx].transAxes,
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    axes[-1].set_xlabel('Time (decoder output)', fontsize=11)
    
    plt.suptitle(f'Nearest Neighbors Analysis: Code {code_idx}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def analyze_why_codes_similar(decoded_codes, characteristics, neighbors, code_idx):
    """
    Analizza PERCHÉ i codici vicini sono simili
    
    Returns:
        analysis: dict con analisi dettagliata
    """
    neighbor_list = neighbors[code_idx]
    
    analysis = {
        'code_idx': code_idx,
        'neighbors': [],
        'similarity_reasons': []
    }
    
    main_trace = decoded_codes[code_idx, 0, :]
    
    # Caratteristiche main code
    main_chars = {
        'amplitude': np.max(main_trace) - np.min(main_trace),
        'mean': np.mean(main_trace),
        'variance': np.var(main_trace),
        'max': np.max(main_trace),
        'min': np.min(main_trace),
        'energy': np.sum(main_trace ** 2)
    }
    
    for neighbor_idx, similarity in neighbor_list:
        neighbor_trace = decoded_codes[neighbor_idx, 0, :]
        
        # Caratteristiche neighbor
        neighbor_chars = {
            'amplitude': np.max(neighbor_trace) - np.min(neighbor_trace),
            'mean': np.mean(neighbor_trace),
            'variance': np.var(neighbor_trace),
            'max': np.max(neighbor_trace),
            'min': np.min(neighbor_trace),
            'energy': np.sum(neighbor_trace ** 2)
        }
        
        # Differenze
        diffs = {k: abs(neighbor_chars[k] - main_chars[k]) for k in main_chars}
        
        # Trova caratteristiche più simili
        # Normalizza le differenze
        normalized_diffs = {}
        for k in diffs:
            if main_chars[k] != 0:
                normalized_diffs[k] = diffs[k] / abs(main_chars[k])
            else:
                normalized_diffs[k] = diffs[k]
        
        # Ordina per differenza (crescente = più simile)
        sorted_features = sorted(normalized_diffs.items(), key=lambda x: x[1])
        
        # Motivi di similarità (top 3 feature più simili)
        reasons = [f"{feat}: Δ{val:.3f}" for feat, val in sorted_features[:3]]
        
        analysis['neighbors'].append({
            'neighbor_idx': neighbor_idx,
            'similarity': similarity,
            'main_chars': main_chars.copy(),
            'neighbor_chars': neighbor_chars.copy(),
            'diffs': diffs.copy(),
            'normalized_diffs': normalized_diffs.copy(),
            'top_similar_features': sorted_features[:3]
        })
        
        analysis['similarity_reasons'].append(reasons)
    
    return analysis


def main():
    print("="*70)
    print("🔬 PCA ANALYSIS DEL CODEBOOK VQ-VAE")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Embedding dim: {MODEL_CONFIG['embedding_dim']}")
    print(f"   PCA components: {N_COMPONENTS_VIZ} (for viz)")
    print(f"   W&B Project: {WANDB_PROJECT}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Inizializza W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "checkpoint_path": CHECKPOINT_PATH,
            "n_components_viz": N_COMPONENTS_VIZ,
            **MODEL_CONFIG
        },
        tags=["pca", "codebook-analysis", "similarity", "clustering"]
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
        
        # 5. Calcola similarità EMBEDDING-SPACE
        print("\n🔍 Calcolando similarità (embedding space)...")
        similarity_embeddings = cosine_similarity(embeddings)
        
        # 6. Calcola similarità DECODED-SPACE
        print("🔍 Calcolando similarità (decoded space)...")
        similarity_decoded = compute_decoded_similarities(decoded_codes)
        
        # 7. PCA
        pca_full, pca_coords_full = perform_pca(embeddings, n_components=N_COMPONENTS_FULL)
        
        # PCA per visualizzazione (2D e 3D)
        pca_viz, pca_coords_viz = perform_pca(embeddings, n_components=N_COMPONENTS_VIZ)
        
        # 8. Visualizzazioni
        print("\n🎨 Creando visualizzazioni...")
        
        # PCA 2D plots
        fig_pca_2d = plot_pca_2d_visualization(pca_coords_viz, characteristics)
        wandb.log({"pca/2d_visualization": wandb.Image(fig_pca_2d)})
        plt.close(fig_pca_2d)
        
        # PCA 3D plots
        if N_COMPONENTS_VIZ >= 3:
            fig_pca_3d = plot_pca_3d_visualization(pca_coords_viz, characteristics)
            wandb.log({"pca/3d_visualization": wandb.Image(fig_pca_3d)})
            plt.close(fig_pca_3d)
        
        # Explained variance
        fig_var = plot_explained_variance(pca_full)
        wandb.log({"pca/explained_variance": wandb.Image(fig_var)})
        plt.close(fig_var)
        
        # PCA loadings
        fig_loadings = plot_pca_loadings(pca_viz, n_top_features=10)
        wandb.log({"pca/loadings": wandb.Image(fig_loadings)})
        plt.close(fig_loadings)
        
        # Heatmap similarità (embedding space)
        fig_sim_emb = plot_similarity_heatmap(similarity_embeddings, 
                                              "Similarity (Embedding Space)")
        wandb.log({"similarity/embedding_space": wandb.Image(fig_sim_emb)})
        plt.close(fig_sim_emb)
        
        # Heatmap similarità (decoded space)
        fig_sim_dec = plot_similarity_heatmap(similarity_decoded,
                                              "Similarity (Decoded Space)")
        wandb.log({"similarity/decoded_space": wandb.Image(fig_sim_dec)})
        plt.close(fig_sim_dec)
        
        # Hierarchical clustering
        fig_cluster = plot_hierarchical_clustering(embeddings, similarity_embeddings)
        wandb.log({"clustering/dendrogram": wandb.Image(fig_cluster)})
        plt.close(fig_cluster)
        
        # 9. Trova nearest neighbors
        print("\n🔍 Trovando nearest neighbors...")
        neighbors = find_nearest_neighbors(similarity_decoded, k=5)
        
        # 10. Visualizza esempi di nearest neighbors
        print("\n🎨 Visualizzando nearest neighbors...")
        
        # Seleziona codici interessanti
        interesting_codes = []
        
        # Max amplitude
        max_amp_code = np.argmax(characteristics['amplitude'])
        interesting_codes.append(('max_amplitude', max_amp_code))
        
        # Min amplitude
        min_amp_code = np.argmin(characteristics['amplitude'])
        interesting_codes.append(('min_amplitude', min_amp_code))
        
        # Median amplitude
        median_amp_code = np.argsort(characteristics['amplitude'])[len(characteristics['amplitude'])//2]
        interesting_codes.append(('median_amplitude', median_amp_code))
        
        # NULL code (7 se esiste)
        if 7 < MODEL_CONFIG['num_embeddings']:
            interesting_codes.append(('null_code', 7))
        
        for label, code_idx in interesting_codes:
            fig_nn = visualize_nearest_neighbors(decoded_codes, neighbors, code_idx, num_neighbors=3)
            wandb.log({f"neighbors/{label}_code{code_idx}": wandb.Image(fig_nn)})
            plt.close(fig_nn)
            
            # Analizza perché sono simili
            analysis = analyze_why_codes_similar(decoded_codes, characteristics, 
                                                 neighbors, code_idx)
            
            print(f"\n   📊 Analysis for Code {code_idx} ({label}):")
            for i, neighbor_info in enumerate(analysis['neighbors'][:3], 1):
                n_idx = neighbor_info['neighbor_idx']
                sim = neighbor_info['similarity']
                top_feats = neighbor_info['top_similar_features']
                
                print(f"      Neighbor {i}: Code {n_idx} (sim={sim:.3f})")
                print(f"         Most similar features:")
                for feat, diff in top_feats:
                    print(f"            - {feat}: normalized_diff={diff:.4f}")
        
        # 11. Log metriche aggregate
        print("\n📊 Loggando metriche aggregate...")
        
        # Statistiche similarità
        mean_sim_emb = np.mean(similarity_embeddings[np.triu_indices(len(similarity_embeddings), k=1)])
        std_sim_emb = np.std(similarity_embeddings[np.triu_indices(len(similarity_embeddings), k=1)])
        
        mean_sim_dec = np.mean(similarity_decoded[np.triu_indices(len(similarity_decoded), k=1)])
        std_sim_dec = np.std(similarity_decoded[np.triu_indices(len(similarity_decoded), k=1)])
        
        # Varianza spiegata dalle prime N componenti
        var_explained_pc1 = pca_full.explained_variance_ratio_[0]
        var_explained_pc2 = pca_full.explained_variance_ratio_[1] if len(pca_full.explained_variance_ratio_) > 1 else 0
        var_explained_pc3 = pca_full.explained_variance_ratio_[2] if len(pca_full.explained_variance_ratio_) > 2 else 0
        var_explained_top3 = np.sum(pca_full.explained_variance_ratio_[:3])
        
        wandb.log({
            "statistics/mean_similarity_embeddings": mean_sim_emb,
            "statistics/std_similarity_embeddings": std_sim_emb,
            "statistics/mean_similarity_decoded": mean_sim_dec,
            "statistics/std_similarity_decoded": std_sim_dec,
            "statistics/mean_amplitude": np.mean(characteristics['amplitude']),
            "statistics/std_amplitude": np.std(characteristics['amplitude']),
            "statistics/mean_variance": np.mean(characteristics['variance']),
            "statistics/mean_num_peaks": np.mean(characteristics['num_peaks']),
            "pca/variance_explained_pc1": var_explained_pc1,
            "pca/variance_explained_pc2": var_explained_pc2,
            "pca/variance_explained_pc3": var_explained_pc3,
            "pca/variance_explained_top3": var_explained_top3,
        })
        
        print(f"\n📊 PCA Statistics:")
        print(f"   PC1 explains: {var_explained_pc1:.3f} of variance")
        print(f"   PC2 explains: {var_explained_pc2:.3f} of variance")
        print(f"   PC3 explains: {var_explained_pc3:.3f} of variance")
        print(f"   Top 3 PCs explain: {var_explained_top3:.3f} of variance")
        
        print("\n" + "="*70)
        print("✅ PCA ANALYSIS COMPLETATA!")
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