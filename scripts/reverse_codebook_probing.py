#!/usr/bin/env python3
"""
Reverse Context Probing: Testa il codice 7 al centro di contesti diversi.

Strategia:
- Codice centrale FISSO: Codice 7
- Contesto VARIABILE: Ogni codice (0-15) ripetuto come contesto
- Sequenze: [X, X, X, X, X, X, X, 7, X, X, X, X, X, X, X] dove X varia

Uso:
    python scripts/codebook_reverse_context_probing.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from datetime import datetime

from models.vqvae import CalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/best_multi_session_FIXED_VQ_16.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-reverse-context-probing"
WANDB_RUN_NAME = f"reverse-context-{datetime.now().strftime('%m%d-%H%M')}"

MODEL_CONFIG = {
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

# ✅ CONFIGURAZIONE ESPERIMENTO - REVERSE
CENTER_CODE_INDEX = 7      # Codice FISSO al centro (il tuo NULL code)
TEMPORAL_LENGTH = 15       # Lunghezza totale sequenza
CENTER_POSITION = 5        # Posizione del codice fisso (metà = 7)
CENTER_WIDTH = 5  

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


def create_reverse_context_sequence(codebook_embeddings, context_code_idx, 
                                    center_code_idx, temporal_length, 
                                    center_position, center_width, device):  # ← AGGIUNGI center_width
    """
    Crea sequenza con MULTIPLI codici centrali consecutivi.
    
    Esempio con center_position=5, center_width=5:
    [X, X, X, X, X, 7, 7, 7, 7, 7, X, X, X, X, X]
                   ↑─────────────↑
                   5 codici 7
    """
    
    # Ottieni embeddings
    context_embedding = codebook_embeddings[context_code_idx]
    center_embedding = codebook_embeddings[center_code_idx]
    
    # Crea sequenza di CONTEXT
    sequence = context_embedding.unsqueeze(1).repeat(1, temporal_length)
    
    # ✅ NUOVO: Sostituisci centro con MULTIPLI codici 7
    for i in range(center_width):
        pos = center_position + i
        if pos < temporal_length:  # Safety check
            sequence[:, pos] = center_embedding
    
    # Aggiungi batch dimension
    sequence = sequence.unsqueeze(0).to(device)
    
    return sequence


def probe_reverse_context_effect(model, center_code_idx, center_position, 
                                 center_width, temporal_length, device):  # ← AGGIUNGI center_width
    """
    Prova tutti i codici come CONTESTO con MULTIPLI codici 7 FISSI al centro.
    """
    
    num_codes = MODEL_CONFIG['num_embeddings']
    codebook_embeddings = model.vector_quantization.embedding.weight.data
    
    print(f"\n🔬 Reverse Context Probing:")
    print(f"   CENTER code (FIXED): {center_code_idx}")
    print(f"   Center positions: {center_position} to {center_position + center_width - 1}")  # ← NUOVO
    print(f"   Center width: {center_width} consecutive codes")  # ← NUOVO
    print(f"   Testing {num_codes} different CONTEXT codes...")
    
    all_reconstructions = []
    
    model.eval()
    with torch.no_grad():
        # 1. BASELINE: Solo CENTER code ripetuto (tutto 7)
        print("\n   🔵 Baseline: Full CENTER sequence (all code 7)...")
        center_embedding = codebook_embeddings[center_code_idx]
        baseline_sequence = center_embedding.unsqueeze(1).repeat(1, temporal_length).unsqueeze(0).to(device)
        baseline_recon = model.decode(baseline_sequence).cpu().numpy()[0]
        
        # 2. REVERSE CONTEXT TEST con multipli codici centrali
        print("\n   🔴 Testing context codes with MULTIPLE CENTER codes...")
        for context_code_idx in range(num_codes):
            # ✅ Passa center_width
            sequence = create_reverse_context_sequence(
                codebook_embeddings, 
                context_code_idx,
                center_code_idx,
                temporal_length, 
                center_position,
                center_width,  # ← AGGIUNGI
                device
            )
            
            reconstruction = model.decode(sequence)
            all_reconstructions.append(reconstruction.cpu().numpy()[0])
            
            if (context_code_idx + 1) % 5 == 0:
                print(f"      Processati {context_code_idx + 1}/{num_codes} context codes...")
    
    reconstructions = np.array(all_reconstructions)
    print(f"\n✅ Reconstructions shape: {reconstructions.shape}")
    print(f"✅ Baseline shape: {baseline_recon.shape}")
    
    return reconstructions, baseline_recon

def log_reverse_context_analysis(reconstructions, baseline_recon, center_code_idx, center_position):
    """
    Analizza e logga i risultati del REVERSE context probing.
    
    Args:
        reconstructions: (num_codes, num_neurons, time)
        baseline_recon: (num_neurons, time)
    """
    
    num_codes, num_neurons, output_time = reconstructions.shape
    
    print(f"\n📊 Analisi Reverse Context Effect...")
    
    # ========================================================================
    # 1. DIFFERENZA DA BASELINE
    # ========================================================================
    print("\n🎨 1/5 - Calcolando differenze da baseline...")
    
    # Calcola differenza per ogni contesto
    differences = reconstructions - baseline_recon[np.newaxis, :, :]  # (num_codes, neurons, time)
    
    # Metrica: MSE differenza
    mse_diff = np.mean(differences ** 2, axis=(1, 2))  # (num_codes,)
    
    # Metrica: Mean absolute difference
    mae_diff = np.mean(np.abs(differences), axis=(1, 2))  # (num_codes,)
    
    # ========================================================================
    # 2. HEATMAP: Differenze Medie
    # ========================================================================
    print("🎨 2/5 - Heatmap differenze...")
    
    mean_diff = np.mean(differences, axis=2)  # (num_codes, neurons)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(mean_diff, aspect='auto', cmap='RdBu_r', 
                   vmin=-1, vmax=1, interpolation='nearest')
    
    ax.set_xlabel('Neurons', fontsize=12)
    ax.set_ylabel('Context Code Index', fontsize=12)
    ax.set_title(f'Reverse Context Effect: Difference from Baseline (CENTER={center_code_idx})\n'
                 f'How different context codes affect fixed center code {center_code_idx}', 
                 fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Δ Activity (vs all-CENTER baseline)')
    plt.tight_layout()
    
    wandb.log({"reverse_context/difference_heatmap": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 3. TOP 5 CONTESTI CON MAGGIORE EFFETTO
    # ========================================================================
    print("🎨 3/5 - Identificando contesti con maggiore effetto...")
    
    # Ordina per MSE differenza
    top_5_effect = np.argsort(mse_diff)[-5:][::-1]
    
    print(f"\n   Top 5 CONTESTI con MAGGIORE effetto sul centro (code {center_code_idx}):")
    for idx in top_5_effect:
        print(f"      Context Code {idx}: MSE diff={mse_diff[idx]:.4f}, MAE diff={mae_diff[idx]:.4f}")
    
    # Plot top 5
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Baseline
    im0 = axes[0].imshow(baseline_recon, aspect='auto', cmap='viridis', vmin=-2, vmax=2)
    axes[0].set_title(f'BASELINE: All CENTER (code {center_code_idx})', fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Neurons')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Top 5
    for i, code_idx in enumerate(top_5_effect, 1):
        im = axes[i].imshow(reconstructions[code_idx], aspect='auto', 
                           cmap='viridis', vmin=-2, vmax=2)
        axes[i].set_title(f'CONTEXT=Code {code_idx}\n(MSE diff={mse_diff[code_idx]:.3f})')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Neurons')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    plt.suptitle(f'Top 5 Context Codes with Strongest Effect on Center (code {center_code_idx})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    wandb.log({"reverse_context/top5_effect": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 4. ANALISI TEMPORALE: Effetto nel tempo
    # ========================================================================
    print("🎨 4/5 - Analisi temporale dell'effetto...")
    
    # Per ogni timestep, calcola quanto cambia l'attività rispetto al baseline
    temporal_effect = np.mean(np.abs(differences), axis=1)  # (num_codes, time)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot per ogni codice
    for code_idx in range(num_codes):
        alpha = 0.3 if code_idx not in top_5_effect else 1.0
        linewidth = 1 if code_idx not in top_5_effect else 2.5
        label = f"Context {code_idx}" if code_idx in top_5_effect else None
        
        ax.plot(temporal_effect[code_idx], alpha=alpha, linewidth=linewidth, label=label)
    
    # Evidenzia centro
    ax.axvline(center_position, color='red', linestyle='--', linewidth=2, 
               label=f'Center Position (code {center_code_idx})')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Mean Absolute Difference', fontsize=12)
    ax.set_title('Temporal Evolution of Reverse Context Effect', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    wandb.log({"reverse_context/temporal_effect": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 5. BAR CHART: MSE Differenza per Codice di Contesto
    # ========================================================================
    print("🎨 5/5 - Bar chart MSE differenze...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['red' if i in top_5_effect else 'skyblue' for i in range(num_codes)]
    
    ax.bar(range(num_codes), mse_diff, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Context Code Index', fontsize=12)
    ax.set_ylabel('MSE Difference from Baseline', fontsize=12)
    ax.set_title(f'Reverse Context Effect: How Each Context Affects Center Code {center_code_idx}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight CENTER code
    ax.axvline(center_code_idx - 0.5, color='green', linestyle='--', linewidth=2, 
               label=f'CENTER Code ({center_code_idx})')
    ax.legend()
    
    plt.tight_layout()
    
    wandb.log({"reverse_context/mse_differences": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 6. CONFRONTO: Self-Context vs Others
    # ========================================================================
    print("🎨 6/6 - Confronto self-context vs others...")
    
    # MSE quando contesto = centro (code 7 ovunque)
    self_context_mse = mse_diff[center_code_idx]
    
    # MSE medio degli altri contesti
    other_contexts_mse = np.delete(mse_diff, center_code_idx)
    other_contexts_mean = np.mean(other_contexts_mse)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Self-Context\n(all code 7)', 'Other Contexts\n(average)']
    values = [self_context_mse, other_contexts_mean]
    colors_bar = ['green', 'orange']
    
    bars = ax.bar(categories, values, color=colors_bar, edgecolor='black', alpha=0.7)
    
    # Aggiungi valori sopra le barre
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('MSE Difference', fontsize=12)
    ax.set_title(f'Self-Context vs Other Contexts Effect on Code {center_code_idx}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    wandb.log({"reverse_context/self_vs_others": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 7. LOG METRICHE
    # ========================================================================
    print("\n📊 Loggando metriche...")
    
    wandb.log({
        "reverse_context/center_code": center_code_idx,
        "reverse_context/center_position": center_position,
        "reverse_context/num_contexts_tested": num_codes,
        "reverse_context/mean_mse_diff": float(np.mean(mse_diff)),
        "reverse_context/max_mse_diff": float(np.max(mse_diff)),
        "reverse_context/min_mse_diff": float(np.min(mse_diff)),
        "reverse_context/self_context_mse": float(self_context_mse),
        "reverse_context/other_contexts_mean_mse": float(other_contexts_mean),
        "reverse_context/top5_contexts": top_5_effect.tolist(),
        "reverse_context/top5_mse_diffs": [float(mse_diff[i]) for i in top_5_effect],
    })
    
    # Tabella con tutti i risultati
    table_data = []
    for i in range(num_codes):
        table_data.append([
            i,                                      # Context code index
            float(mse_diff[i]),                     # MSE diff
            float(mae_diff[i]),                     # MAE diff
            "TOP 5" if i in top_5_effect else "",   # Top 5 marker
            "SELF" if i == center_code_idx else "", # Self-context marker
        ])
    
    table = wandb.Table(
        columns=["Context Code", "MSE Diff", "MAE Diff", "Top 5", "Self"],
        data=table_data
    )
    
    wandb.log({"reverse_context/results_table": table})
    
    print("\n✅ Analisi reverse context effect completata!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🔬 REVERSE CONTEXT PROBING: Multiple Center Codes")  # ← CAMBIA titolo
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num codes: {MODEL_CONFIG['num_embeddings']}")
    print(f"   CENTER code (FIXED): {CENTER_CODE_INDEX}")
    print(f"   Center positions: {CENTER_POSITION} to {CENTER_POSITION + CENTER_WIDTH - 1}")  # ← NUOVO
    print(f"   Center width: {CENTER_WIDTH} consecutive codes")  # ← NUOVO
    print(f"   W&B Project: {WANDB_PROJECT}")
    print(f"\n🎯 Esperimento:")
    print(f"   Testa come CONTESTI diversi influenzano {CENTER_WIDTH} codici {CENTER_CODE_INDEX} consecutivi")  # ← NUOVO
    print(f"   Sequenze: [X, X, X, X, X, 7, 7, 7, 7, 7, X, X, X, X, X]")  # ← AGGIORNA
    print(f"   dove X varia da 0 a {MODEL_CONFIG['num_embeddings']-1}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Inizializza W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "checkpoint_path": CHECKPOINT_PATH,
            "center_code": CENTER_CODE_INDEX,
            "center_position": CENTER_POSITION,
            "center_width": CENTER_WIDTH,  # ← AGGIUNGI
            "temporal_length": TEMPORAL_LENGTH,
            "experiment_type": "reverse_context_multiple",  # ← AGGIORNA
            **MODEL_CONFIG
        },
        tags=["reverse-context-probing", "multiple-center", "code-7-center"]  # ← AGGIORNA
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # Carica modello
        model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # ✅ AGGIUNGI center_width alla chiamata
        reconstructions, baseline_recon = probe_reverse_context_effect(
            model, 
            CENTER_CODE_INDEX, 
            CENTER_POSITION,
            CENTER_WIDTH,  # ← AGGIUNGI
            TEMPORAL_LENGTH, 
            device
        )
        
        # Analizza (log_reverse_context_analysis rimane uguale)
        log_reverse_context_analysis(
            reconstructions, 
            baseline_recon, 
            CENTER_CODE_INDEX, 
            CENTER_POSITION
        )
        
        print("\n" + "="*70)
        print("✅ REVERSE CONTEXT PROBING COMPLETATO!")
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