#!/usr/bin/env python3
"""
t-SNE Analysis del Codebook VQ-VAE

Analizza la struttura del codebook attraverso:
1. t-SNE visualization dei codici
2. Heatmap di similarità (cosine similarity)
3. Clustering gerarchico
4. Analisi dei codici vicini (perché sono simili?)
5. Decodifica dei codici per capire cosa rappresentano

Uso:
    python scripts/codebook_tsne_analysis.py
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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from models.vqvae import CalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/best_single_neuron_model_32.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-codebook-tsne-analysis"
WANDB_RUN_NAME = f"tsne-analysis_32codes"

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

# t-SNE parameters
TSNE_PERPLEXITY = 5  # Per 32 codici, perplexity bassa
TSNE_N_ITER = 3000
TSNE_RANDOM_STATE = 42

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


def perform_tsne(embeddings, perplexity=30, n_iter=3000, random_state=42):
    """Esegue t-SNE sugli embeddings"""
    print(f"\n🔬 Eseguendo t-SNE...")
    print(f"   Perplexity: {perplexity}")
    print(f"   Iterations: {n_iter}")
    
    # Prima PCA per ridurre dimensionalità (opzionale ma consigliato)
    if embeddings.shape[1] > 50:
        print(f"   Pre-processing con PCA (50 componenti)...")
        pca = PCA(n_components=50)
        embeddings_reduced = pca.fit_transform(embeddings)
        print(f"   Varianza spiegata: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        embeddings_reduced = embeddings
    
    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1
    )
    
    tsne_coords = tsne.fit_transform(embeddings_reduced)
    print(f"✅ t-SNE completato: {tsne_coords.shape}")
    
    return tsne_coords


def plot_tsne_visualization(tsne_coords, characteristics, similarity_matrix):
    """
    Crea visualizzazioni t-SNE con diverse colorazioni
    
    Args:
        tsne_coords: (num_codes, 2) coordinate t-SNE
        characteristics: dict con caratteristiche dei codici
        similarity_matrix: matrice di similarità
    """
    num_codes = tsne_coords.shape[0]
    
    # ========================================================================
    # FIGURA 1: t-SNE con multiple colorazioni
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('t-SNE Visualization del Codebook', fontsize=16, fontweight='bold')
    
    # 1. Colored by code index
    scatter1 = axes[0, 0].scatter(tsne_coords[:, 0], tsne_coords[:, 1], 
                                  c=range(num_codes), cmap='tab20', s=100, alpha=0.7)
    for i in range(num_codes):
        axes[0, 0].annotate(str(i), (tsne_coords[i, 0], tsne_coords[i, 1]),
                           fontsize=8, ha='center', va='center')
    axes[0, 0].set_title('Colored by Code Index')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Code Index')
    
    # 2. Colored by amplitude
    scatter2 = axes[0, 1].scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                                  c=characteristics['amplitude'], cmap='viridis', s=100, alpha=0.7)
    axes[0, 1].set_title('Colored by Amplitude')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Amplitude')
    
    # 3. Colored by mean activity
    scatter3 = axes[0, 2].scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                                  c=characteristics['mean_activity'], cmap='coolwarm', s=100, alpha=0.7)
    axes[0, 2].set_title('Colored by Mean Activity')
    axes[0, 2].set_xlabel('t-SNE 1')
    axes[0, 2].set_ylabel('t-SNE 2')
    plt.colorbar(scatter3, ax=axes[0, 2], label='Mean Activity')
    
    # 4. Colored by variance
    scatter4 = axes[1, 0].scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                                  c=characteristics['variance'], cmap='plasma', s=100, alpha=0.7)
    axes[1, 0].set_title('Colored by Variance')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter4, ax=axes[1, 0], label='Variance')
    
    # 5. Colored by num_peaks
    scatter5 = axes[1, 1].scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                                  c=characteristics['num_peaks'], cmap='Spectral', s=100, alpha=0.7)
    axes[1, 1].set_title('Colored by Number of Peaks')
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter5, ax=axes[1, 1], label='Num Peaks')
    
    # 6. Colored by energy
    scatter6 = axes[1, 2].scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                                  c=characteristics['energy'], cmap='magma', s=100, alpha=0.7)
    axes[1, 2].set_title('Colored by Energy')
    axes[1, 2].set_xlabel('t-SNE 1')
    axes[1, 2].set_ylabel('t-SNE 2')
    plt.colorbar(scatter6, ax=axes[1, 2], label='Energy')
    
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
    print("🔬 t-SNE ANALYSIS DEL CODEBOOK VQ-VAE")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Embedding dim: {MODEL_CONFIG['embedding_dim']}")
    print(f"   t-SNE perplexity: {TSNE_PERPLEXITY}")
    print(f"   W&B Project: {WANDB_PROJECT}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Inizializza W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "checkpoint_path": CHECKPOINT_PATH,
            "tsne_perplexity": TSNE_PERPLEXITY,
            "tsne_n_iter": TSNE_N_ITER,
            **MODEL_CONFIG
        },
        tags=["tsne", "codebook-analysis", "similarity", "clustering"]
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
        
        # 7. t-SNE
        tsne_coords = perform_tsne(embeddings, 
                                   perplexity=TSNE_PERPLEXITY,
                                   n_iter=TSNE_N_ITER,
                                   random_state=TSNE_RANDOM_STATE)
        
        # 8. Visualizzazioni
        print("\n🎨 Creando visualizzazioni...")
        
        # t-SNE plots
        fig_tsne = plot_tsne_visualization(tsne_coords, characteristics, similarity_embeddings)
        wandb.log({"tsne/visualization": wandb.Image(fig_tsne)})
        plt.close(fig_tsne)
        
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
        
        # Seleziona alcuni codici interessanti
        # - Codice con massima ampiezza
        # - Codice con minima ampiezza
        # - Codice "medio"
        # - Codice NULL (7 se esiste)
        
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
        
        wandb.log({
            "statistics/mean_similarity_embeddings": mean_sim_emb,
            "statistics/std_similarity_embeddings": std_sim_emb,
            "statistics/mean_similarity_decoded": mean_sim_dec,
            "statistics/std_similarity_decoded": std_sim_dec,
            "statistics/mean_amplitude": np.mean(characteristics['amplitude']),
            "statistics/std_amplitude": np.std(characteristics['amplitude']),
            "statistics/mean_variance": np.mean(characteristics['variance']),
            "statistics/mean_num_peaks": np.mean(characteristics['num_peaks']),
        })
        
        print("\n" + "="*70)
        print("✅ t-SNE ANALYSIS COMPLETATA!")
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