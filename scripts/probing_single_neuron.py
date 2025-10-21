#!/usr/bin/env python3
"""
Context Probing per SINGOLO NEURONE: Testa l'effetto del contesto sul codebook.

Strategia:
- Codice "nullo" (baseline): Codice 7
- Crea sequenze: [7, 7, 7, X, 7, 7, 7] dove X varia
- Vedi come il codice centrale X influenza la ricostruzione del singolo neurone

Uso:
    python scripts/codebook_probing_single_neuron.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
from scipy.stats import pearsonr

from models.vqvae import CalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/best_single_neuron_model.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-context-probing-single-neuron"
WANDB_RUN_NAME = f"context-probe-1neuron-{datetime.now().strftime('%m%d-%H%M')}"

MODEL_CONFIG = {
    'num_neurons': 1,              # ✅ 1 neurone
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 32,
    'embedding_dim': 32,
    'commitment_cost': 0.25,
    'dropout_rate': 0.1,
    'use_quantizer': True,
}

# ✅ CONFIGURAZIONE ESPERIMENTO
NULL_CODE_INDEX = 7        # Codice "nullo" di base (baseline)
TEMPORAL_LENGTH = 15       # Lunghezza totale sequenza
CENTER_POSITION = 7        # Posizione del codice variabile (metà = 7)

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


def create_context_sequence(codebook_embeddings, null_code_idx, center_code_idx, 
                           temporal_length, center_position, device):
    """
    Crea sequenza: [NULL, NULL, ..., CENTER, ..., NULL, NULL]
    
    Args:
        codebook_embeddings: Tensor (num_codes, embedding_dim)
        null_code_idx: Indice del codice "nullo"
        center_code_idx: Indice del codice centrale
        temporal_length: Lunghezza totale (es. 15)
        center_position: Posizione del codice centrale (es. 7)
        device: torch device
    
    Returns:
        sequence: Tensor (1, embedding_dim, temporal_length)
    """
    
    # Ottieni embeddings
    null_embedding = codebook_embeddings[null_code_idx]    # (embedding_dim,)
    center_embedding = codebook_embeddings[center_code_idx]  # (embedding_dim,)
    
    # Crea sequenza di NULL
    sequence = null_embedding.unsqueeze(1).repeat(1, temporal_length)  # (embedding_dim, temporal_length)
    
    # Sostituisci centro con CENTER code
    sequence[:, center_position] = center_embedding
    
    # Aggiungi batch dimension
    sequence = sequence.unsqueeze(0).to(device)  # (1, embedding_dim, temporal_length)
    
    return sequence


def probe_context_effect(model, null_code_idx, center_position, temporal_length, device):
    """
    Prova tutti i codici come "centro" in un contesto di NULL codes.
    
    Returns:
        reconstructions: Array (num_codes, 1, output_time)
        baseline_recon: Array (1, output_time) - solo NULL
    """
    
    num_codes = MODEL_CONFIG['num_embeddings']
    codebook_embeddings = model.vector_quantization.embedding.weight.data
    
    print(f"\n🔬 Context Probing:")
    print(f"   NULL code: {null_code_idx}")
    print(f"   Center position: {center_position}/{temporal_length}")
    print(f"   Testing {num_codes} different center codes...")
    
    all_reconstructions = []
    
    model.eval()
    with torch.no_grad():
        # 1. BASELINE: Solo NULL codes (nessun centro variabile)
        print("\n   🔵 Baseline: Full NULL sequence...")
        null_embedding = codebook_embeddings[null_code_idx]
        baseline_sequence = null_embedding.unsqueeze(1).repeat(1, temporal_length).unsqueeze(0).to(device)
        baseline_recon = model.decode(baseline_sequence).cpu().numpy()[0]  # Shape: (1, output_time)
        
        # 2. CONTEXT TEST: Varia il codice centrale
        print("\n   🔴 Testing center codes...")
        for center_code_idx in range(num_codes):
            # Crea sequenza con questo codice al centro
            sequence = create_context_sequence(
                codebook_embeddings, 
                null_code_idx, 
                center_code_idx,
                temporal_length, 
                center_position, 
                device
            )
            
            # Decodifica
            reconstruction = model.decode(sequence)
            all_reconstructions.append(reconstruction.cpu().numpy()[0])  # Shape: (1, output_time)
            
            if (center_code_idx + 1) % 5 == 0:
                print(f"      Processati {center_code_idx + 1}/{num_codes} center codes...")
    
    reconstructions = np.array(all_reconstructions)  # Shape: (num_codes, 1, output_time)
    print(f"\n✅ Reconstructions shape: {reconstructions.shape}")
    print(f"✅ Baseline shape: {baseline_recon.shape}")
    
    return reconstructions, baseline_recon


def visualize_trace_comparison(reconstructions, baseline_recon, code_indices, 
                               null_code_idx, title_prefix=""):
    """
    Visualizza tracce temporali per confronto (invece di heatmap)
    
    Args:
        reconstructions: (num_codes, 1, time)
        baseline_recon: (1, time)
        code_indices: lista di codici da visualizzare
        null_code_idx: indice del codice NULL
        title_prefix: prefisso per il titolo
    """
    
    num_codes_to_show = len(code_indices)
    output_time = reconstructions.shape[2]
    
    fig, axes = plt.subplots(num_codes_to_show + 1, 1, figsize=(14, 3 * (num_codes_to_show + 1)))
    
    if num_codes_to_show == 0:
        axes = [axes]
    
    time_points = np.arange(output_time)
    
    # === BASELINE ===
    axes[0].plot(time_points, baseline_recon[0, :], 'gray', linewidth=2, 
                 label=f'BASELINE (all code {null_code_idx})', alpha=0.8)
    axes[0].set_ylabel('Activity')
    axes[0].set_title(f'{title_prefix}BASELINE: Full NULL (code {null_code_idx})', 
                      fontweight='bold', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, output_time - 1])
    
    # === CENTER CODES ===
    for plot_idx, code_idx in enumerate(code_indices, 1):
        
        # Traccia ricostruita con questo center code
        recon_trace = reconstructions[code_idx, 0, :]
        
        # Plot baseline (grigio) + questo code (colorato)
        axes[plot_idx].plot(time_points, baseline_recon[0, :], 'gray', 
                           linewidth=1, alpha=0.4, label='Baseline')
        axes[plot_idx].plot(time_points, recon_trace, 'b-', 
                           linewidth=2, label=f'CENTER=Code {code_idx}', alpha=0.8)
        
        # Calcola differenza
        diff = recon_trace - baseline_recon[0, :]
        
        # Evidenzia differenze positive/negative
        axes[plot_idx].fill_between(time_points, baseline_recon[0, :], recon_trace,
                                    where=(diff > 0), color='green', alpha=0.2, 
                                    label='Increased activity')
        axes[plot_idx].fill_between(time_points, baseline_recon[0, :], recon_trace,
                                    where=(diff < 0), color='red', alpha=0.2, 
                                    label='Decreased activity')
        
        # Calcola metriche
        mse_diff = np.mean(diff ** 2)
        mae_diff = np.mean(np.abs(diff))
        
        if np.std(baseline_recon[0, :]) > 1e-8 and np.std(recon_trace) > 1e-8:
            corr, _ = pearsonr(baseline_recon[0, :], recon_trace)
        else:
            corr = 1.0
        
        axes[plot_idx].set_ylabel('Activity')
        axes[plot_idx].set_title(f'CENTER=Code {code_idx} | '
                                 f'MSE diff={mse_diff:.4f}, Corr={corr:.3f}',
                                 fontweight='bold', fontsize=11)
        axes[plot_idx].legend(loc='upper right', fontsize=9)
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_xlim([0, output_time - 1])
        
        # Statistiche testuali
        stats_text = f'Max diff: {np.max(np.abs(diff)):.3f}\n'
        stats_text += f'Mean |diff|: {mae_diff:.3f}'
        axes[plot_idx].text(0.02, 0.98, stats_text, 
                           transform=axes[plot_idx].transAxes,
                           verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    axes[-1].set_xlabel('Time (decoder output)', fontsize=11)
    
    plt.tight_layout()
    
    return fig


def log_context_analysis(reconstructions, baseline_recon, null_code_idx, center_position):
    """
    Analizza e logga i risultati del context probing con TRACCE.
    
    Args:
        reconstructions: (num_codes, 1, time)
        baseline_recon: (1, time)
    """
    
    num_codes = reconstructions.shape[0]
    output_time = reconstructions.shape[2]
    
    print(f"\n📊 Analisi Context Effect...")
    
    # ========================================================================
    # 1. CALCOLA DIFFERENZE DA BASELINE
    # ========================================================================
    print("\n🎨 1/5 - Calcolando differenze da baseline...")
    
    # Calcola differenza per ogni codice
    differences = reconstructions[:, 0, :] - baseline_recon[0, :]  # (num_codes, time)
    
    # Metrica: MSE differenza
    mse_diff = np.mean(differences ** 2, axis=1)  # (num_codes,)
    
    # Metrica: Mean absolute difference
    mae_diff = np.mean(np.abs(differences), axis=1)  # (num_codes,)
    
    # ========================================================================
    # 2. VISUALIZZA TOP 5 CODICI CON MAGGIORE EFFETTO
    # ========================================================================
    print("🎨 2/5 - Visualizzando top 5 codici con maggiore effetto...")
    
    # Ordina per MSE differenza
    top_5_effect = np.argsort(mse_diff)[-5:][::-1]
    
    print(f"\n   Top 5 codici con MAGGIORE effetto sul contesto:")
    for idx in top_5_effect:
        print(f"      Code {idx}: MSE diff={mse_diff[idx]:.4f}, MAE diff={mae_diff[idx]:.4f}")
    
    # Plot top 5 come TRACCE
    fig = visualize_trace_comparison(
        reconstructions, 
        baseline_recon, 
        top_5_effect,
        null_code_idx,
        title_prefix="Context Effect - "
    )
    
    wandb.log({"context/top5_effect_traces": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 3. HEATMAP: Overview di TUTTI i codici
    # ========================================================================
    print("🎨 3/5 - Heatmap overview...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap delle ricostruzioni
    im1 = axes[0].imshow(reconstructions[:, 0, :], aspect='auto', cmap='viridis', 
                        vmin=-2, vmax=2, interpolation='nearest')
    axes[0].set_xlabel('Time', fontsize=11)
    axes[0].set_ylabel('Center Code Index', fontsize=11)
    axes[0].set_title(f'All Reconstructions (NULL={null_code_idx})', 
                     fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Activity')
    
    # Evidenzia NULL code
    axes[0].axhline(y=null_code_idx, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'NULL code ({null_code_idx})')
    axes[0].legend(loc='upper right')
    
    # Heatmap delle differenze
    im2 = axes[1].imshow(differences, aspect='auto', cmap='RdBu_r', 
                        vmin=-1, vmax=1, interpolation='nearest')
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].set_ylabel('Center Code Index', fontsize=11)
    axes[1].set_title(f'Difference from Baseline', 
                     fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Δ Activity')
    
    # Evidenzia NULL code
    axes[1].axhline(y=null_code_idx, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'NULL code ({null_code_idx})')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    wandb.log({"context/overview_heatmap": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 4. ANALISI TEMPORALE: Effetto nel tempo
    # ========================================================================
    print("🎨 4/5 - Analisi temporale dell'effetto...")
    
    # Per ogni timestep, calcola quanto cambia l'attività
    temporal_effect = np.abs(differences)  # (num_codes, time)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot per ogni codice
    for code_idx in range(num_codes):
        alpha = 0.3 if code_idx not in top_5_effect else 1.0
        linewidth = 1 if code_idx not in top_5_effect else 2.5
        label = f"Code {code_idx}" if code_idx in top_5_effect else None
        
        ax.plot(temporal_effect[code_idx], alpha=alpha, linewidth=linewidth, label=label)
    
    # Evidenzia centro (posizione scalata all'output del decoder)
    # Il decoder potrebbe upsamplare, quindi scala la posizione
    center_scaled = int(center_position * (output_time / TEMPORAL_LENGTH))
    ax.axvline(center_scaled, color='red', linestyle='--', linewidth=2, 
               label=f'Center Position (scaled)')
    
    ax.set_xlabel('Time (decoder output)', fontsize=12)
    ax.set_ylabel('Absolute Difference', fontsize=12)
    ax.set_title('Temporal Evolution of Context Effect', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    wandb.log({"context/temporal_effect": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 5. BAR CHART: MSE Differenza per Codice
    # ========================================================================
    print("🎨 5/5 - Bar chart MSE differenze...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['red' if i in top_5_effect else 'skyblue' for i in range(num_codes)]
    
    ax.bar(range(num_codes), mse_diff, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Center Code Index', fontsize=12)
    ax.set_ylabel('MSE Difference from Baseline', fontsize=12)
    ax.set_title(f'Context Effect Strength per Code (NULL baseline = {null_code_idx})', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight NULL code
    ax.axvline(null_code_idx - 0.5, color='green', linestyle='--', linewidth=2, 
               label=f'NULL Code ({null_code_idx})')
    ax.legend()
    
    plt.tight_layout()
    
    wandb.log({"context/mse_differences": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 6. LOG METRICHE
    # ========================================================================
    print("\n📊 Loggando metriche...")
    
    wandb.log({
        "context/null_code": null_code_idx,
        "context/center_position": center_position,
        "context/num_codes_tested": num_codes,
        "context/mean_mse_diff": float(np.mean(mse_diff)),
        "context/max_mse_diff": float(np.max(mse_diff)),
        "context/min_mse_diff": float(np.min(mse_diff)),
        "context/top5_codes": top_5_effect.tolist(),
        "context/top5_mse_diffs": [float(mse_diff[i]) for i in top_5_effect],
    })
    
    # # Tabella con tutti i risultati
    # table_data = []
    # for i in range(num_codes):
    #     table_data.append([
    #         i,
    #         float(mse_diff[i]),
    #         float(mae_diff[i]),
    #         "TOP 5" if i in top_5_effect else "",
    #         "NULL" if i == null_code_idx else ""
    #     ])
    # 
    # table = wandb.Table(
    #     columns=["Code Index", "MSE Diff", "MAE Diff", "Top 5", "NULL"],
    #     data=table_data
    # )
    # 
    # wandb.log({"context/results_table": table})
    
    print("\n✅ Analisi context effect completata!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🔬 CONTEXT PROBING: Singolo Neurone (Tracce Temporali)")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num codes: {MODEL_CONFIG['num_embeddings']}")
    print(f"   NULL code: {NULL_CODE_INDEX}")
    print(f"   Center position: {CENTER_POSITION}/{TEMPORAL_LENGTH}")
    print(f"   W&B Project: {WANDB_PROJECT}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Inizializza W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "checkpoint_path": CHECKPOINT_PATH,
            "null_code": NULL_CODE_INDEX,
            "center_position": CENTER_POSITION,
            "temporal_length": TEMPORAL_LENGTH,
            **MODEL_CONFIG
        },
        tags=["context-probing", "single-neuron", "traces", "code-interaction"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # Carica modello
        model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # Esegui context probing
        reconstructions, baseline_recon = probe_context_effect(
            model, 
            NULL_CODE_INDEX, 
            CENTER_POSITION, 
            TEMPORAL_LENGTH, 
            device
        )
        
        # Analizza e logga risultati
        log_context_analysis(
            reconstructions, 
            baseline_recon, 
            NULL_CODE_INDEX, 
            CENTER_POSITION
        )
        
        print("\n" + "="*70)
        print("✅ CONTEXT PROBING COMPLETATO!")
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