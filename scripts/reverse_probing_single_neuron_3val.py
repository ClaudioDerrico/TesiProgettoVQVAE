#!/usr/bin/env python3
"""
Reverse Context Probing per SINGOLO NEURONE: Testa il codice 7 al centro di contesti diversi.

MODIFICATO: Usa 3 CODICI CENTRALI CONSECUTIVI invece di 5

Strategia:
- Codice centrale FISSO: Codice 7 (ripetuto 3 volte)
- Contesto VARIABILE: Ogni codice (0-31) ripetuto come contesto
- Sequenze: [X, X, X, X, X, X, 7, 7, 7, X, X, X, X, X, X] dove X varia

Visualizzazione: TRACCE TEMPORALI del singolo neurone

Uso:
    python reverse_probing_3_codes.py
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

CHECKPOINT_PATH = "results/best_single_neuron_model_32.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-reverse-context-probing-single-neuron"
WANDB_RUN_NAME = f"reverse-context-11codes-16"

MODEL_CONFIG = {
    'num_neurons': 1,              # ✅ 1 neurone
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 16,
    'embedding_dim': 32,
    'commitment_cost': 0.25,
    'dropout_rate': 0.1,
    'use_quantizer': True,
}

# ✅ CONFIGURAZIONE ESPERIMENTO - 3 CODICI CENTRALI
CENTER_CODE_INDEX = 7      # Codice FISSO al centro
TEMPORAL_LENGTH = 15       # Lunghezza totale sequenza (INVARIATA)
CENTER_POSITION = 2        # Posizione iniziale del blocco centrale (per centrare 3 codici in una sequenza di 15)
CENTER_WIDTH = 11           # ✅ 11 codici centrali consecutivi

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
                                    center_position, center_width, device):
    """
    Crea sequenza con MULTIPLI codici centrali consecutivi.
    
    Con center_position=6, center_width=3, temporal_length=15:
    [X, X, X, X, X, X, 7, 7, 7, X, X, X, X, X, X]
                      ↑─────↑
                      3 codici 7
    
    Args:
        codebook_embeddings: Tensor (num_codes, embedding_dim)
        context_code_idx: Indice del codice di contesto (ripetuto)
        center_code_idx: Indice del codice centrale (fisso)
        temporal_length: Lunghezza totale (es. 15)
        center_position: Posizione iniziale del blocco centrale (es. 6)
        center_width: Larghezza del blocco centrale (es. 3)
        device: torch device
    
    Returns:
        sequence: Tensor (1, embedding_dim, temporal_length)
    """
    
    # Ottieni embeddings
    context_embedding = codebook_embeddings[context_code_idx]
    center_embedding = codebook_embeddings[center_code_idx]
    
    # Crea sequenza di CONTEXT
    sequence = context_embedding.unsqueeze(1).repeat(1, temporal_length)  # (embedding_dim, temporal_length)
    
    # ✅ Sostituisci centro con MULTIPLI codici 7
    for i in range(center_width):
        pos = center_position + i
        if pos < temporal_length:  # Safety check
            sequence[:, pos] = center_embedding
    
    # Aggiungi batch dimension
    sequence = sequence.unsqueeze(0).to(device)  # (1, embedding_dim, temporal_length)
    
    return sequence


def probe_reverse_context_effect(model, center_code_idx, center_position, 
                                 center_width, temporal_length, device):
    """
    Prova tutti i codici come CONTESTO con MULTIPLI codici 7 FISSI al centro.
    
    Returns:
        reconstructions: Array (num_codes, 1, output_time)
        baseline_recon: Array (1, output_time) - solo CENTER
    """
    
    num_codes = MODEL_CONFIG['num_embeddings']
    codebook_embeddings = model.vector_quantization.embedding.weight.data
    
    print(f"\n🔬 Reverse Context Probing:")
    print(f"   CENTER code (FIXED): {center_code_idx}")
    print(f"   Center positions: {center_position} to {center_position + center_width - 1}")
    print(f"   Center width: {center_width} consecutive codes")
    print(f"   Total sequence length: {temporal_length}")
    print(f"   Testing {num_codes} different CONTEXT codes...")
    
    all_reconstructions = []
    
    model.eval()
    with torch.no_grad():
        # 1. BASELINE: Solo CENTER code ripetuto (tutto 7)
        print("\n   🔵 Baseline: Full CENTER sequence (all code 7)...")
        center_embedding = codebook_embeddings[center_code_idx]
        baseline_sequence = center_embedding.unsqueeze(1).repeat(1, temporal_length).unsqueeze(0).to(device)
        baseline_recon = model.decode(baseline_sequence).cpu().numpy()[0]  # Shape: (1, output_time)
        
        # 2. REVERSE CONTEXT TEST con multipli codici centrali
        print("\n   🔴 Testing context codes with MULTIPLE CENTER codes...")
        for context_code_idx in range(num_codes):
            # Crea sequenza con questo contesto
            sequence = create_reverse_context_sequence(
                codebook_embeddings, 
                context_code_idx,
                center_code_idx,
                temporal_length, 
                center_position,
                center_width,
                device
            )
            
            # Decodifica
            reconstruction = model.decode(sequence)
            all_reconstructions.append(reconstruction.cpu().numpy()[0])  # Shape: (1, output_time)
            
            if (context_code_idx + 1) % 5 == 0:
                print(f"      Processati {context_code_idx + 1}/{num_codes} context codes...")
    
    reconstructions = np.array(all_reconstructions)  # Shape: (num_codes, 1, output_time)
    print(f"\n✅ Reconstructions shape: {reconstructions.shape}")
    print(f"✅ Baseline shape: {baseline_recon.shape}")
    
    return reconstructions, baseline_recon


def visualize_reverse_trace_comparison(reconstructions, baseline_recon, code_indices, 
                                       center_code_idx, title_prefix=""):
    """
    Visualizza tracce temporali per reverse probing
    
    Args:
        reconstructions: (num_codes, 1, time)
        baseline_recon: (1, time)
        code_indices: lista di codici CONTEXT da visualizzare
        center_code_idx: indice del codice CENTER (fisso)
        title_prefix: prefisso per il titolo
    """
    
    num_codes_to_show = len(code_indices)
    output_time = reconstructions.shape[2]
    
    fig, axes = plt.subplots(num_codes_to_show + 1, 1, figsize=(14, 3 * (num_codes_to_show + 1)))
    
    if num_codes_to_show == 0:
        axes = [axes]
    
    time_points = np.arange(output_time)
    
    # === BASELINE ===
    axes[0].plot(time_points, baseline_recon[0, :], 'green', linewidth=2, 
                 label=f'BASELINE (all code {center_code_idx})', alpha=0.8)
    axes[0].set_ylabel('Activity')
    axes[0].set_title(f'{title_prefix}BASELINE: All CENTER (code {center_code_idx})', 
                      fontweight='bold', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, output_time - 1])
    
    # === CONTEXT CODES ===
    for plot_idx, context_code_idx in enumerate(code_indices, 1):
        
        # Traccia ricostruita con questo context code
        recon_trace = reconstructions[context_code_idx, 0, :]
        
        # Plot baseline (verde) + questo context (blu)
        axes[plot_idx].plot(time_points, baseline_recon[0, :], 'green', 
                           linewidth=1, alpha=0.4, label='Baseline (all center)')
        axes[plot_idx].plot(time_points, recon_trace, 'b-', 
                           linewidth=2, label=f'CONTEXT=Code {context_code_idx}', alpha=0.8)
        
        # Calcola differenza
        diff = recon_trace - baseline_recon[0, :]
        
        # Evidenzia differenze positive/negative
        axes[plot_idx].fill_between(time_points, baseline_recon[0, :], recon_trace,
                                    where=(diff > 0), color='orange', alpha=0.2, 
                                    label='Increased activity')
        axes[plot_idx].fill_between(time_points, baseline_recon[0, :], recon_trace,
                                    where=(diff < 0), color='purple', alpha=0.2, 
                                    label='Decreased activity')
        
        # Calcola metriche
        mse_diff = np.mean(diff ** 2)
        mae_diff = np.mean(np.abs(diff))
        
        if np.std(baseline_recon[0, :]) > 1e-8 and np.std(recon_trace) > 1e-8:
            corr, _ = pearsonr(baseline_recon[0, :], recon_trace)
        else:
            corr = 1.0
        
        # Marker speciale se context == center
        if context_code_idx == center_code_idx:
            title_extra = " ⭐ SELF-CONTEXT"
        else:
            title_extra = ""
        
        axes[plot_idx].set_ylabel('Activity')
        axes[plot_idx].set_title(f'CONTEXT=Code {context_code_idx}{title_extra} | '
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


def log_reverse_context_analysis(reconstructions, baseline_recon, center_code_idx, center_position):
    """
    Analizza e logga i risultati del REVERSE context probing con TRACCE.
    
    Args:
        reconstructions: (num_codes, 1, time)
        baseline_recon: (1, time)
    """
    
    num_codes = reconstructions.shape[0]
    output_time = reconstructions.shape[2]
    
    print(f"\n📊 Analisi Reverse Context Effect...")
    
    # ========================================================================
    # 1. CALCOLA DIFFERENZE DA BASELINE
    # ========================================================================
    print("\n🎨 1/5 - Calcolando differenze da baseline...")
    
    # Calcola differenza per ogni contesto
    differences = reconstructions[:, 0, :] - baseline_recon[0, :]  # (num_codes, time)
    
    # Metrica: MSE differenza
    mse_diff = np.mean(differences ** 2, axis=1)  # (num_codes,)
    
    # Metrica: Mean absolute difference
    mae_diff = np.mean(np.abs(differences), axis=1)  # (num_codes,)
    
    # ========================================================================
    # 2. VISUALIZZA TOP 5 CONTESTI CON MAGGIORE EFFETTO
    # ========================================================================
    print("🎨 2/5 - Visualizzando top 5 contesti con maggiore effetto...")
    
    # Ordina per MSE differenza
    top_5_effect = np.argsort(mse_diff)[-5:][::-1]
    
    print(f"\n   Top 5 CONTESTI con MAGGIORE effetto sul centro (code {center_code_idx}):")
    for idx in top_5_effect:
        print(f"      Context Code {idx}: MSE diff={mse_diff[idx]:.4f}, MAE diff={mae_diff[idx]:.4f}")
    
    # Plot top 5 come TRACCE
    fig = visualize_reverse_trace_comparison(
        reconstructions, 
        baseline_recon, 
        top_5_effect,
        center_code_idx,
        title_prefix="Reverse Context Effect - "
    )
    
    wandb.log({"reverse_context/top5_effect_traces": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 3. HEATMAP: Overview di TUTTI i contesti
    # ========================================================================
    print("🎨 3/5 - Heatmap overview...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap delle ricostruzioni
    im1 = axes[0].imshow(reconstructions[:, 0, :], aspect='auto', cmap='viridis', 
                        vmin=-2, vmax=2, interpolation='nearest')
    axes[0].set_xlabel('Time', fontsize=11)
    axes[0].set_ylabel('Context Code Index', fontsize=11)
    axes[0].set_title(f'All Reconstructions (CENTER={center_code_idx})', 
                     fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Activity')
    
    # Evidenzia CENTER code (self-context)
    axes[0].axhline(y=center_code_idx, color='green', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'CENTER code ({center_code_idx})')
    axes[0].legend(loc='upper right')
    
    # Heatmap delle differenze
    im2 = axes[1].imshow(differences, aspect='auto', cmap='RdBu_r', 
                        vmin=-1, vmax=1, interpolation='nearest')
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].set_ylabel('Context Code Index', fontsize=11)
    axes[1].set_title(f'Difference from Baseline', 
                     fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Δ Activity')
    
    # Evidenzia CENTER code
    axes[1].axhline(y=center_code_idx, color='green', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'CENTER code ({center_code_idx})')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    wandb.log({"reverse_context/overview_heatmap": wandb.Image(fig)})
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
        label = f"Context {code_idx}" if code_idx in top_5_effect else None
        
        ax.plot(temporal_effect[code_idx], alpha=alpha, linewidth=linewidth, label=label)
    
    # Evidenzia zona centrale (posizione scalata all'output del decoder)
    center_start_scaled = int(center_position * (output_time / TEMPORAL_LENGTH))
    center_end_scaled = int((center_position + CENTER_WIDTH) * (output_time / TEMPORAL_LENGTH))
    ax.axvspan(center_start_scaled, center_end_scaled, color='yellow', alpha=0.2, 
               label=f'Center Zone (code {center_code_idx})')
    
    ax.set_xlabel('Time (decoder output)', fontsize=12)
    ax.set_ylabel('Absolute Difference', fontsize=12)
    ax.set_title('Temporal Evolution of Reverse Context Effect', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    wandb.log({"reverse_context/temporal_effect": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 5. BAR CHART + CONFRONTO SELF vs OTHERS
    # ========================================================================
    print("🎨 5/5 - Bar chart MSE differenze + self-context comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # === BAR CHART ===
    colors = ['red' if i in top_5_effect else 'skyblue' for i in range(num_codes)]
    colors[center_code_idx] = 'green'  # Highlight self-context
    
    axes[0].bar(range(num_codes), mse_diff, color=colors, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Context Code Index', fontsize=12)
    axes[0].set_ylabel('MSE Difference from Baseline', fontsize=12)
    axes[0].set_title(f'Reverse Context Effect: How Each Context Affects Center Code {center_code_idx}', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Highlight CENTER code
    axes[0].axvline(center_code_idx - 0.5, color='green', linestyle='--', linewidth=2, 
                   label=f'CENTER Code ({center_code_idx})')
    axes[0].legend()
    
    # === SELF vs OTHERS COMPARISON ===
    self_context_mse = mse_diff[center_code_idx]
    other_contexts_mse = np.delete(mse_diff, center_code_idx)
    other_contexts_mean = np.mean(other_contexts_mse)
    
    categories = [f'Self-Context\n(all code {center_code_idx})', 'Other Contexts\n(average)']
    values = [self_context_mse, other_contexts_mean]
    colors_bar = ['green', 'orange']
    
    bars = axes[1].bar(categories, values, color=colors_bar, edgecolor='black', alpha=0.7)
    
    # Aggiungi valori sopra le barre
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    axes[1].set_ylabel('MSE Difference', fontsize=12)
    axes[1].set_title(f'Self-Context vs Other Contexts Effect on Code {center_code_idx}', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    wandb.log({"reverse_context/mse_differences_and_comparison": wandb.Image(fig)})
    plt.close(fig)
    
    # ========================================================================
    # 6. LOG METRICHE
    # ========================================================================
    print("\n📊 Loggando metriche...")
    
    wandb.log({
        "reverse_context/center_code": center_code_idx,
        "reverse_context/center_position": center_position,
        "reverse_context/center_width": CENTER_WIDTH,
        "reverse_context/num_contexts_tested": num_codes,
        "reverse_context/mean_mse_diff": float(np.mean(mse_diff)),
        "reverse_context/max_mse_diff": float(np.max(mse_diff)),
        "reverse_context/min_mse_diff": float(np.min(mse_diff)),
        "reverse_context/self_context_mse": float(self_context_mse),
        "reverse_context/other_contexts_mean_mse": float(other_contexts_mean),
        "reverse_context/top5_contexts": top_5_effect.tolist(),
        "reverse_context/top5_mse_diffs": [float(mse_diff[i]) for i in top_5_effect],
    })
    
    print("\n✅ Analisi reverse context effect completata!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🔬 REVERSE CONTEXT PROBING: 3 Codici Centrali (Singolo Neurone)")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num codes: {MODEL_CONFIG['num_embeddings']}")
    print(f"   CENTER code (FIXED): {CENTER_CODE_INDEX}")
    print(f"   Center positions: {CENTER_POSITION} to {CENTER_POSITION + CENTER_WIDTH - 1}")
    print(f"   Center width: {CENTER_WIDTH} consecutive codes")
    print(f"   Total sequence length: {TEMPORAL_LENGTH}")
    print(f"   W&B Project: {WANDB_PROJECT}")
    print(f"\n🎯 Esperimento:")
    print(f"   Testa come CONTESTI diversi influenzano {CENTER_WIDTH} codici {CENTER_CODE_INDEX} consecutivi")
    print(f"   Sequenze: [X, X, X, X, X, X, 7, 7, 7, X, X, X, X, X, X]")
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
            "center_width": CENTER_WIDTH,
            "temporal_length": TEMPORAL_LENGTH,
            "experiment_type": "reverse_context_3_codes",
            **MODEL_CONFIG
        },
        tags=["reverse-context-probing", "single-neuron", "traces", "3-center-codes", "code-7-center"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # Carica modello
        model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # Esegui reverse context probing
        reconstructions, baseline_recon = probe_reverse_context_effect(
            model, 
            CENTER_CODE_INDEX, 
            CENTER_POSITION,
            CENTER_WIDTH,
            TEMPORAL_LENGTH, 
            device
        )
        
        # Analizza e logga risultati
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