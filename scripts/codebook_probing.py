#!/usr/bin/env python3
"""
Script per fare CODEBOOK PROBING con logging SOLO su W&B.
NON crea file locali, tutto viene loggato su W&B.

Uso:
    python scripts/codebook_probing.py
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

from models.vqvae import CalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/best_multi_session_FIXED_VQ_128.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-codebook-probing"
WANDB_RUN_NAME = f"probe-128codes-{datetime.now().strftime('%m%d-%H%M')}"

MODEL_CONFIG = {
    'num_neurons': 30,
    'num_hiddens': 512,
    'num_residual_layers': 6,
    'num_residual_hiddens': 256,
    'num_embeddings': 128,      # I tuoi 128 codici
    'embedding_dim': 128,        # Dimensione embedding
    'commitment_cost': 0.5,
    'dropout_rate': 0.0,
    'use_quantizer': True,
}

# Lunghezza sequenza temporale (deve matchare il tuo window_size dopo encoding)
# Se window_size=60 e hai 2 downsampling con stride=2, ottieni ~15 timesteps
TEMPORAL_LENGTH = 15  # Adatta in base al tuo modello

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


def probe_all_codes(model, num_codes, temporal_length, device):
    """
    Prova tutti i codici del codebook.
    
    Args:
        model: Modello VQ-VAE caricato
        num_codes: Numero totale di codici (128)
        temporal_length: Lunghezza sequenza temporale
        device: Device
    
    Returns:
        reconstructions: Array (num_codes, num_neurons, output_time)
    """
    
    print(f"\n🔬 Probing {num_codes} codici del codebook...")
    print(f"   Temporal length: {temporal_length}")
    
    # Accedi al codebook embeddings
    codebook_embeddings = model.vector_quantization.embedding.weight.data
    print(f"   Codebook shape: {codebook_embeddings.shape}")
    
    all_reconstructions = []
    
    model.eval()
    with torch.no_grad():
        for code_idx in range(num_codes):
            # Ottieni l'embedding per questo codice
            # Shape: (embedding_dim,)
            code_embedding = codebook_embeddings[code_idx]
            
            # Crea sequenza ripetuta nel tempo
            # Shape: (embedding_dim, temporal_length)
            repeated_sequence = code_embedding.unsqueeze(1).repeat(1, temporal_length)
            
            # Aggiungi batch dimension
            # Shape: (1, embedding_dim, temporal_length)
            repeated_sequence = repeated_sequence.unsqueeze(0).to(device)
            
            # Passa al decoder
            reconstruction = model.decode(repeated_sequence)
            
            # Shape: (1, num_neurons, output_time)
            all_reconstructions.append(reconstruction.cpu().numpy()[0])
            
            if (code_idx + 1) % 20 == 0:
                print(f"   Processati {code_idx + 1}/{num_codes} codici...")
    
    reconstructions = np.array(all_reconstructions)
    print(f"✅ Reconstructions shape: {reconstructions.shape}")
    
    return reconstructions


def log_to_wandb(reconstructions):
    """
    Analizza e logga TUTTO su W&B (zero file locali).
    
    Args:
        reconstructions: Array (num_codes, num_neurons, time)
    """
    
    num_codes, num_neurons, output_time = reconstructions.shape
    
    print(f"\n📊 Loggando su W&B...")
    print(f"   Shape: {reconstructions.shape}")
    
    # ========================================================================
    # 1. HEATMAP GLOBALE: Tutti i codici x tutti i neuroni (media temporale)
    # ========================================================================
    print("\n🎨 1/6 - Heatmap globale...")
    
    # Media temporale per ogni codice-neurone
    # Shape: (num_codes, num_neurons)
    mean_activity = np.mean(reconstructions, axis=2)
    
    fig, ax = plt.subplots(figsize=(12, 16))
    
    im = ax.imshow(mean_activity, aspect='auto', cmap='RdBu_r', 
                   vmin=-2, vmax=2, interpolation='nearest')
    
    ax.set_xlabel('Neurons', fontsize=12)
    ax.set_ylabel('Codebook Index', fontsize=12)
    ax.set_title(f'Codebook Probing: Mean Activity per Code\n({num_codes} codes × {num_neurons} neurons)', 
                 fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Mean ΔF/F')
    plt.tight_layout()
    
    wandb.log({"codebook/heatmap_global": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 2. ESEMPI INDIVIDUALI: Primi 16 codici
    # ========================================================================
    print("🎨 2/6 - Esempi individuali (primi 16)...")
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    for i in range(min(16, num_codes)):
        code_recon = reconstructions[i]
        
        im = axes[i].imshow(code_recon, aspect='auto', cmap='viridis',
                           vmin=-2, vmax=2)
        axes[i].set_title(f'Code {i}', fontsize=10)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Neurons')
        
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    plt.suptitle('Codebook Probing: First 16 Codes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    wandb.log({"codebook/examples_first16": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 3. LOG OGNI SINGOLO CODICE (per esplorazione dettagliata)
    # ========================================================================
    print("🎨 3/6 - Loggando ogni singolo codice...")
    
    # Log primi 32 codici individualmente (per non sovraccaricare W&B)
    for code_idx in range(min(32, num_codes)):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(reconstructions[code_idx], aspect='auto', 
                      cmap='viridis', vmin=-2, vmax=2)
        ax.set_title(f'Code {code_idx}', fontsize=12)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Neurons', fontsize=10)
        plt.colorbar(im, ax=ax, label='ΔF/F')
        
        plt.tight_layout()
        
        wandb.log({f"codebook/individual/code_{code_idx:03d}": wandb.Image(fig)})
        plt.close(fig)
    
    # ========================================================================
    # 4. CLUSTERING: Raggruppa codici simili
    # ========================================================================
    print("🔍 4/6 - Clustering dei codici...")
    
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    
    # Flatten: (num_codes, num_neurons * time)
    flat_reconstructions = reconstructions.reshape(num_codes, -1)
    
    # PCA per visualizzazione
    pca = PCA(n_components=2)
    codes_2d = pca.fit_transform(flat_reconstructions)
    
    # KMeans clustering
    n_clusters = 8  # Numero di cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(flat_reconstructions)
    
    # Plot PCA con colori per cluster
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(codes_2d[:, 0], codes_2d[:, 1], 
                        c=cluster_labels, cmap='tab10', 
                        s=100, alpha=0.7, edgecolors='black')
    
    # Annota ogni punto con il suo indice
    for i, (x, y) in enumerate(codes_2d):
        ax.text(x, y, str(i), fontsize=8, ha='center', va='center')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_title(f'Codebook Clustering (K-means, k={n_clusters})', 
                 fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    
    wandb.log({"codebook/clustering_pca": wandb.Image(fig)})
    plt.close(fig)
    
    # Log cluster assignments
    wandb.log({"codebook/cluster_distribution": wandb.Histogram(cluster_labels)})
    
    # ========================================================================
    # 5. STATISTICHE PER CODICE
    # ========================================================================
    print("📈 5/6 - Statistiche per codice...")
    
    mean_per_code = np.mean(reconstructions, axis=(1, 2))
    std_per_code = np.std(reconstructions, axis=(1, 2))
    max_per_code = np.max(reconstructions, axis=(1, 2))
    min_per_code = np.min(reconstructions, axis=(1, 2))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mean
    axes[0, 0].bar(range(num_codes), mean_per_code, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Code Index')
    axes[0, 0].set_ylabel('Mean Activity')
    axes[0, 0].set_title('Mean Activity per Code')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std
    axes[0, 1].bar(range(num_codes), std_per_code, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Code Index')
    axes[0, 1].set_ylabel('Std Activity')
    axes[0, 1].set_title('Std Activity per Code')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Max
    axes[1, 0].bar(range(num_codes), max_per_code, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Code Index')
    axes[1, 0].set_ylabel('Max Activity')
    axes[1, 0].set_title('Max Activity per Code')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Min
    axes[1, 1].bar(range(num_codes), min_per_code, color='lightyellow', edgecolor='black')
    axes[1, 1].set_xlabel('Code Index')
    axes[1, 1].set_ylabel('Min Activity')
    axes[1, 1].set_title('Min Activity per Code')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Statistics per Codebook Entry', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    wandb.log({"codebook/statistics": wandb.Image(fig)})
    plt.close(fig)
    
    # Log distribuzioni come histogram
    wandb.log({
        "codebook/mean_distribution": wandb.Histogram(mean_per_code),
        "codebook/std_distribution": wandb.Histogram(std_per_code),
    })
    
    # ========================================================================
    # 6. CODICI ESTREMI
    # ========================================================================
    print("🔍 6/6 - Codici estremi...")
    
    # Codici con attività più alta/bassa
    top_5_high = np.argsort(mean_per_code)[-5:][::-1]
    top_5_low = np.argsort(mean_per_code)[:5]
    
    print(f"\n   Top 5 codici con ALTA attività:")
    for idx in top_5_high:
        print(f"      Code {idx}: mean={mean_per_code[idx]:.4f}")
    
    print(f"\n   Top 5 codici con BASSA attività:")
    for idx in top_5_low:
        print(f"      Code {idx}: mean={mean_per_code[idx]:.4f}")
    
    # Visualizza i 5 più estremi
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # Top 5 high
    for i, code_idx in enumerate(top_5_high):
        im = axes[0, i].imshow(reconstructions[code_idx], aspect='auto', 
                              cmap='viridis', vmin=-2, vmax=2)
        axes[0, i].set_title(f'HIGH: Code {code_idx}\n(mean={mean_per_code[code_idx]:.3f})')
        axes[0, i].set_xlabel('Time')
        axes[0, i].set_ylabel('Neurons')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)
    
    # Top 5 low
    for i, code_idx in enumerate(top_5_low):
        im = axes[1, i].imshow(reconstructions[code_idx], aspect='auto', 
                              cmap='viridis', vmin=-2, vmax=2)
        axes[1, i].set_title(f'LOW: Code {code_idx}\n(mean={mean_per_code[code_idx]:.3f})')
        axes[1, i].set_xlabel('Time')
        axes[1, i].set_ylabel('Neurons')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)
    
    plt.suptitle('Extreme Codes: Top 5 High vs Low Activity', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    wandb.log({"codebook/extremes": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 7. LOG METRICHE AGGREGATE
    # ========================================================================
    print("\n📊 Loggando metriche aggregate...")
    
    # Crea tabella con statistiche per ogni codice
    table_data = []
    for i in range(num_codes):
        table_data.append([
            i,                      # Code index
            cluster_labels[i],      # Cluster
            mean_per_code[i],       # Mean
            std_per_code[i],        # Std
            max_per_code[i],        # Max
            min_per_code[i],        # Min
        ])
    
    table = wandb.Table(
        columns=["Code Index", "Cluster", "Mean", "Std", "Max", "Min"],
        data=table_data
    )
    
    wandb.log({"codebook/statistics_table": table})
    
    # Log metriche scalari
    wandb.log({
        "codebook/mean_activity_overall": float(np.mean(mean_per_code)),
        "codebook/std_activity_overall": float(np.std(mean_per_code)),
        "codebook/max_activity_overall": float(np.max(mean_per_code)),
        "codebook/min_activity_overall": float(np.min(mean_per_code)),
        "codebook/num_clusters": n_clusters,
        "codebook/top5_high_indices": top_5_high.tolist(),
        "codebook/top5_low_indices": top_5_low.tolist(),
    })
    
    # ========================================================================
    # 8. SIMILARITY MATRIX (bonus)
    # ========================================================================
    print("\n🎨 Bonus - Similarity matrix...")
    
    # Calcola similarità coseno tra tutti i codici
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity_matrix = cosine_similarity(flat_reconstructions)
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    im = ax.imshow(similarity_matrix, aspect='auto', cmap='coolwarm', 
                   vmin=-1, vmax=1)
    
    ax.set_xlabel('Code Index', fontsize=12)
    ax.set_ylabel('Code Index', fontsize=12)
    ax.set_title(f'Codebook Similarity Matrix (Cosine)\n({num_codes} × {num_codes})', 
                 fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    
    wandb.log({"codebook/similarity_matrix": wandb.Image(fig)})
    plt.close(fig)
    
    print("\n✅ Tutto loggato su W&B!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🔬 CODEBOOK PROBING: Analisi Completa su W&B")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num codes: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Embedding dim: {MODEL_CONFIG['embedding_dim']}")
    print(f"   Temporal length: {TEMPORAL_LENGTH}")
    print(f"   W&B Project: {WANDB_PROJECT}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Inizializza W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "checkpoint_path": CHECKPOINT_PATH,
            "temporal_length": TEMPORAL_LENGTH,
            **MODEL_CONFIG
        },
        tags=["codebook-probing", "semantic-analysis"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # Carica modello
        model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # Prova tutti i codici
        reconstructions = probe_all_codes(
            model, 
            MODEL_CONFIG['num_embeddings'], 
            TEMPORAL_LENGTH, 
            device
        )
        
        # Logga tutto su W&B (zero file locali)
        log_to_wandb(reconstructions)
        
        print("\n" + "="*70)
        print("✅ CODEBOOK PROBING COMPLETATO!")
        print(f"   Tutti i risultati su W&B: {wandb.run.url}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Chiudi W&B run
        wandb.finish()
        print("✅ W&B run completato!")


if __name__ == "__main__":
    main()