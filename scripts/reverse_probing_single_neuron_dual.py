#!/usr/bin/env python3
"""
Reverse Context Probing DUAL CODEBOOK per SINGOLO NEURONE: Testa il codice 7 al centro di contesti diversi.

Strategia:
- Codice centrale FISSO: Codice 7 per entrambi i codebook
- Contesto VARIABILE: Ogni codice (0-N) ripetuto come contesto
- Sequenze: [X, X, X, X, X, X, X, 7, X, X, X, X, X, X, X] dove X varia
- Analizza separatamente per ACTIVE e INACTIVE codebook

Visualizzazione: TRACCE TEMPORALI del singolo neurone

Uso:
    python scripts/reverse_probing_single_neuron_dual.py
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

from models.dual_vqvae import DualCalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/dual_codebook/best_dual_model_32.pth"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-reverse-context-probing-single-neuron-dual"
WANDB_RUN_NAME = f"reverse-context-1neuron-dual-{datetime.now().strftime('%m%d-%H%M')}"

MODEL_CONFIG = {
    'num_neurons': 1,              # ✅ 1 neurone
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 32, 
    'embedding_dim': 32,
    'commitment_cost': 0.5,
    'dropout_rate': 0.0,
    'use_quantizer': True,
    'threshold': 0.0,
    'threshold_type': 'adaptive',
}

# ✅ CONFIGURAZIONE ESPERIMENTO - REVERSE
CENTER_CODE_INDEX = 7      # Codice FISSO al centro (il tuo NULL code)
TEMPORAL_LENGTH = 15       # Lunghezza totale sequenza
CENTER_POSITION = 5        # Posizione del codice fisso (inizio della zona centrale)
CENTER_WIDTH = 5           # Larghezza della zona centrale (5 codici consecutivi)

# ============================================================================
# FUNZIONI
# ============================================================================

def load_model(checkpoint_path, config, device):
    """Carica il modello dual allenato"""
    print(f"🔄 Caricando modello DUAL da: {checkpoint_path}")
    
    model = DualCalciumVQVAE(
        num_neurons=config['num_neurons'],
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        dropout_rate=config['dropout_rate'],
        use_quantizer=config['use_quantizer'],
        threshold=config.get('threshold', 0.0),
        threshold_type=config.get('threshold_type', 'adaptive')
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Auto-infer num_embeddings se necessario
    if 'model_config' in checkpoint:
        model_config_checkpoint = checkpoint['model_config']
        if 'num_embeddings' in model_config_checkpoint:
            config['num_embeddings'] = model_config_checkpoint['num_embeddings']
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Modello DUAL caricato (epoca {checkpoint.get('epoch', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Modello DUAL caricato")
    
    model = model.to(device)
    model.eval()
    
    return model


def create_reverse_context_sequence(codebook_embeddings, context_code_idx, 
                                    center_code_idx, temporal_length, 
                                    center_position, center_width, device):
    """
    Crea sequenza con MULTIPLI codici centrali consecutivi.
    
    Esempio con center_position=5, center_width=5:
    [X, X, X, X, X, 7, 7, 7, 7, 7, X, X, X, X, X]
                   ↑─────────────↑
                   5 codici 7
    
    Args:
        codebook_embeddings: Tensor (num_codes, embedding_dim)
        context_code_idx: Indice del codice di contesto (ripetuto)
        center_code_idx: Indice del codice centrale (fisso)
        temporal_length: Lunghezza totale (es. 15)
        center_position: Posizione iniziale del blocco centrale (es. 5)
        center_width: Larghezza del blocco centrale (es. 5)
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


def probe_reverse_context_effect_dual(model, center_code_idx, center_position, 
                                     center_width, temporal_length, device):
    """
    Prova tutti i codici come CONTESTO con MULTIPLI codici 7 FISSI al centro.
    Esegue per entrambi i codebook (active e inactive).
    
    Returns:
        reconstructions_active: Array (num_codes_per_book, 1, output_time)
        reconstructions_inactive: Array (num_codes_per_book, 1, output_time)
        baseline_recon_active: Array (1, output_time) - solo CENTER
        baseline_recon_inactive: Array (1, output_time) - solo CENTER
    """
    
    vq = model.vector_quantization
    n_e_per_codebook = vq.n_e_per_codebook  # Numero embeddings per codebook
    
    # Verifica che CENTER_CODE_INDEX sia valido
    if center_code_idx >= n_e_per_codebook:
        raise ValueError(f"CENTER_CODE_INDEX ({center_code_idx}) deve essere < n_e_per_codebook ({n_e_per_codebook})")
    
    # Ottieni embeddings dei due codebook
    embeddings_active = vq.embedding_active.weight.data
    embeddings_inactive = vq.embedding_inactive.weight.data
    
    print(f"\n🔬 Reverse Context Probing DUAL:")
    print(f"   CENTER code (FIXED): {center_code_idx}")
    print(f"   Center positions: {center_position} to {center_position + center_width - 1}")
    print(f"   Center width: {center_width} consecutive codes")
    print(f"   Active codebook: {n_e_per_codebook} codes (indici 0-{n_e_per_codebook-1})")
    print(f"   Inactive codebook: {n_e_per_codebook} codes (indici 0-{n_e_per_codebook-1})")
    print(f"   Testing {n_e_per_codebook} different CONTEXT codes per codebook...")
    
    all_reconstructions_active = []
    all_reconstructions_inactive = []
    
    model.eval()
    with torch.no_grad():
        # ====================================================================
        # ACTIVE CODEBOOK
        # ====================================================================
        print("\n   🔴 ACTIVE Codebook:")
        
        # BASELINE: Solo CENTER code ripetuto (tutto 7)
        print("      🔵 Baseline: Full CENTER sequence (all code 7)...")
        center_embedding_active = embeddings_active[center_code_idx]
        baseline_sequence_active = center_embedding_active.unsqueeze(1).repeat(1, temporal_length).unsqueeze(0).to(device)
        baseline_recon_active = model.decode(baseline_sequence_active).cpu().numpy()[0]  # Shape: (1, output_time)
        
        # REVERSE CONTEXT TEST con multipli codici centrali
        print("      🔴 Testing context codes with MULTIPLE CENTER codes...")
        for context_code_idx in range(n_e_per_codebook):
            sequence = create_reverse_context_sequence(
                embeddings_active, 
                context_code_idx,
                center_code_idx,
                temporal_length, 
                center_position,
                center_width,
                device
            )
            reconstruction = model.decode(sequence)
            all_reconstructions_active.append(reconstruction.cpu().numpy()[0])  # Shape: (1, output_time)
            
            if (context_code_idx + 1) % 50 == 0:
                print(f"         Processati {context_code_idx + 1}/{n_e_per_codebook} context codes...")
        
        # ====================================================================
        # INACTIVE CODEBOOK
        # ====================================================================
        print("\n   🔵 INACTIVE Codebook:")
        
        # BASELINE: Solo CENTER code ripetuto (tutto 7)
        print("      🔵 Baseline: Full CENTER sequence (all code 7)...")
        center_embedding_inactive = embeddings_inactive[center_code_idx]
        baseline_sequence_inactive = center_embedding_inactive.unsqueeze(1).repeat(1, temporal_length).unsqueeze(0).to(device)
        baseline_recon_inactive = model.decode(baseline_sequence_inactive).cpu().numpy()[0]  # Shape: (1, output_time)
        
        # REVERSE CONTEXT TEST con multipli codici centrali
        print("      🔴 Testing context codes with MULTIPLE CENTER codes...")
        for context_code_idx in range(n_e_per_codebook):
            sequence = create_reverse_context_sequence(
                embeddings_inactive, 
                context_code_idx,
                center_code_idx,
                temporal_length, 
                center_position,
                center_width,
                device
            )
            reconstruction = model.decode(sequence)
            all_reconstructions_inactive.append(reconstruction.cpu().numpy()[0])  # Shape: (1, output_time)
            
            if (context_code_idx + 1) % 50 == 0:
                print(f"         Processati {context_code_idx + 1}/{n_e_per_codebook} context codes...")
    
    reconstructions_active = np.array(all_reconstructions_active)  # Shape: (num_codes, 1, output_time)
    reconstructions_inactive = np.array(all_reconstructions_inactive)  # Shape: (num_codes, 1, output_time)
    
    print(f"\n✅ Active reconstructions shape: {reconstructions_active.shape}")
    print(f"✅ Inactive reconstructions shape: {reconstructions_inactive.shape}")
    print(f"✅ Active baseline shape: {baseline_recon_active.shape}")
    print(f"✅ Inactive baseline shape: {baseline_recon_inactive.shape}")
    
    return (reconstructions_active, reconstructions_inactive,
            baseline_recon_active, baseline_recon_inactive)


def visualize_reverse_trace_comparison_dual(reconstructions_active, reconstructions_inactive,
                                           baseline_recon_active, baseline_recon_inactive,
                                           code_indices, center_code_idx, title_prefix=""):
    """
    Visualizza tracce temporali per reverse probing DUAL
    
    Args:
        reconstructions_active: (num_codes, 1, time)
        reconstructions_inactive: (num_codes, 1, time)
        baseline_recon_active: (1, time)
        baseline_recon_inactive: (1, time)
        code_indices: lista di codici CONTEXT da visualizzare
        center_code_idx: indice del codice CENTER (fisso)
        title_prefix: prefisso per il titolo
    """
    
    num_codes_to_show = len(code_indices)
    output_time = reconstructions_active.shape[2]
    
    # Crea subplot: una riga per ACTIVE, una per INACTIVE, per ogni codice
    n_cols = num_codes_to_show + 1  # +1 per baseline
    
    fig, axes = plt.subplots(2, n_cols, figsize=(14 * n_cols, 10))
    
    # Gestisci il caso in cui axes è 1D quando n_cols=1
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(2, n_cols)
    
    time_points = np.arange(output_time)
    
    # === ACTIVE BASELINE ===
    axes[0, 0].plot(time_points, baseline_recon_active[0, :], 'green', linewidth=2, 
                   label=f'BASELINE (all code {center_code_idx})', alpha=0.8)
    axes[0, 0].set_ylabel('Activity')
    axes[0, 0].set_title(f'{title_prefix}ACTIVE BASELINE: All CENTER (code {center_code_idx})', 
                         fontweight='bold', fontsize=12)
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, output_time - 1])
    
    # === INACTIVE BASELINE ===
    axes[1, 0].plot(time_points, baseline_recon_inactive[0, :], 'green', linewidth=2, 
                   label=f'BASELINE (all code {center_code_idx})', alpha=0.8)
    axes[1, 0].set_ylabel('Activity')
    axes[1, 0].set_title(f'{title_prefix}INACTIVE BASELINE: All CENTER (code {center_code_idx})', 
                         fontweight='bold', fontsize=12)
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, output_time - 1])
    
    # === CONTEXT CODES ===
    for plot_idx, context_code_idx in enumerate(code_indices, 1):
        
        # ACTIVE
        recon_trace_active = reconstructions_active[context_code_idx, 0, :]
        axes[0, plot_idx].plot(time_points, baseline_recon_active[0, :], 'green', 
                              linewidth=1, alpha=0.4, label='Baseline (all center)')
        axes[0, plot_idx].plot(time_points, recon_trace_active, 'b-', 
                              linewidth=2, label=f'CONTEXT=Code {context_code_idx}', alpha=0.8)
        
        diff_active = recon_trace_active - baseline_recon_active[0, :]
        axes[0, plot_idx].fill_between(time_points, baseline_recon_active[0, :], recon_trace_active,
                                      where=(diff_active > 0), color='orange', alpha=0.2, 
                                      label='Increased activity')
        axes[0, plot_idx].fill_between(time_points, baseline_recon_active[0, :], recon_trace_active,
                                      where=(diff_active < 0), color='purple', alpha=0.2, 
                                      label='Decreased activity')
        
        mse_diff_active = np.mean(diff_active ** 2)
        mae_diff_active = np.mean(np.abs(diff_active))
        if np.std(baseline_recon_active[0, :]) > 1e-8 and np.std(recon_trace_active) > 1e-8:
            corr_active, _ = pearsonr(baseline_recon_active[0, :], recon_trace_active)
        else:
            corr_active = 1.0
        
        if context_code_idx == center_code_idx:
            title_extra_active = " ⭐ SELF-CONTEXT"
        else:
            title_extra_active = ""
        
        axes[0, plot_idx].set_ylabel('Activity')
        axes[0, plot_idx].set_title(f'ACTIVE: CONTEXT=Code {context_code_idx}{title_extra_active} | '
                                    f'MSE diff={mse_diff_active:.4f}, Corr={corr_active:.3f}',
                                    fontweight='bold', fontsize=11)
        axes[0, plot_idx].legend(loc='upper right', fontsize=9)
        axes[0, plot_idx].grid(True, alpha=0.3)
        axes[0, plot_idx].set_xlim([0, output_time - 1])
        
        # INACTIVE
        recon_trace_inactive = reconstructions_inactive[context_code_idx, 0, :]
        axes[1, plot_idx].plot(time_points, baseline_recon_inactive[0, :], 'green', 
                              linewidth=1, alpha=0.4, label='Baseline (all center)')
        axes[1, plot_idx].plot(time_points, recon_trace_inactive, 'b-', 
                              linewidth=2, label=f'CONTEXT=Code {context_code_idx}', alpha=0.8)
        
        diff_inactive = recon_trace_inactive - baseline_recon_inactive[0, :]
        axes[1, plot_idx].fill_between(time_points, baseline_recon_inactive[0, :], recon_trace_inactive,
                                      where=(diff_inactive > 0), color='orange', alpha=0.2, 
                                      label='Increased activity')
        axes[1, plot_idx].fill_between(time_points, baseline_recon_inactive[0, :], recon_trace_inactive,
                                      where=(diff_inactive < 0), color='purple', alpha=0.2, 
                                      label='Decreased activity')
        
        mse_diff_inactive = np.mean(diff_inactive ** 2)
        mae_diff_inactive = np.mean(np.abs(diff_inactive))
        if np.std(baseline_recon_inactive[0, :]) > 1e-8 and np.std(recon_trace_inactive) > 1e-8:
            corr_inactive, _ = pearsonr(baseline_recon_inactive[0, :], recon_trace_inactive)
        else:
            corr_inactive = 1.0
        
        if context_code_idx == center_code_idx:
            title_extra_inactive = " ⭐ SELF-CONTEXT"
        else:
            title_extra_inactive = ""
        
        axes[1, plot_idx].set_ylabel('Activity')
        axes[1, plot_idx].set_title(f'INACTIVE: CONTEXT=Code {context_code_idx}{title_extra_inactive} | '
                                    f'MSE diff={mse_diff_inactive:.4f}, Corr={corr_inactive:.3f}',
                                    fontweight='bold', fontsize=11)
        axes[1, plot_idx].legend(loc='upper right', fontsize=9)
        axes[1, plot_idx].grid(True, alpha=0.3)
        axes[1, plot_idx].set_xlim([0, output_time - 1])
    
    # Imposta xlabel sull'ultima colonna (se esiste più di una colonna)
    if n_cols > 1:
        axes[0, -1].set_xlabel('Time (decoder output)', fontsize=11)
        axes[1, -1].set_xlabel('Time (decoder output)', fontsize=11)
    else:
        axes[0, 0].set_xlabel('Time (decoder output)', fontsize=11)
        axes[1, 0].set_xlabel('Time (decoder output)', fontsize=11)
    
    plt.tight_layout()
    
    return fig


def log_reverse_context_analysis_dual(reconstructions_active, reconstructions_inactive,
                                     baseline_recon_active, baseline_recon_inactive,
                                     center_code_idx, center_position):
    """
    Analizza e logga i risultati del REVERSE context probing con TRACCE per entrambi i codebook.
    
    Args:
        reconstructions_active: (num_codes, 1, time)
        reconstructions_inactive: (num_codes, 1, time)
        baseline_recon_active: (1, time)
        baseline_recon_inactive: (1, time)
    """
    
    num_codes = reconstructions_active.shape[0]
    output_time = reconstructions_active.shape[2]
    
    print(f"\n📊 Analisi Reverse Context Effect DUAL...")
    
    # ========================================================================
    # 1. CALCOLA DIFFERENZE DA BASELINE
    # ========================================================================
    print("\n🎨 Calcolando differenze da baseline...")
    
    # Active
    differences_active = reconstructions_active[:, 0, :] - baseline_recon_active[0, :]  # (num_codes, time)
    mse_diff_active = np.mean(differences_active ** 2, axis=1)  # (num_codes,)
    mae_diff_active = np.mean(np.abs(differences_active), axis=1)  # (num_codes,)
    
    # Inactive
    differences_inactive = reconstructions_inactive[:, 0, :] - baseline_recon_inactive[0, :]  # (num_codes, time)
    mse_diff_inactive = np.mean(differences_inactive ** 2, axis=1)  # (num_codes,)
    mae_diff_inactive = np.mean(np.abs(differences_inactive), axis=1)  # (num_codes,)
    
    # ========================================================================
    # 2. VISUALIZZA TOP 2 CONTESTI PER CODEBOOK
    # ========================================================================
    print("🎨 Visualizzando top 2 contesti per codebook...")
    
    # Trova i top 2 contesti con maggiore effetto per ACTIVE e INACTIVE
    top_2_active = np.argsort(mse_diff_active)[-2:][::-1]  # Top 2, ordinati dal maggiore
    top_2_inactive = np.argsort(mse_diff_inactive)[-2:][::-1]  # Top 2, ordinati dal maggiore
    
    print(f"\n   Top 2 ACTIVE contesti:")
    for i, ctx_idx in enumerate(top_2_active, 1):
        print(f"      {i}. Context Code {ctx_idx}: MSE diff={mse_diff_active[ctx_idx]:.4f}, MAE diff={mae_diff_active[ctx_idx]:.4f}")
    
    print(f"\n   Top 2 INACTIVE contesti:")
    for i, ctx_idx in enumerate(top_2_inactive, 1):
        print(f"      {i}. Context Code {ctx_idx}: MSE diff={mse_diff_inactive[ctx_idx]:.4f}, MAE diff={mae_diff_inactive[ctx_idx]:.4f}")
    
    # Combina top 2 di entrambi i codebook (rimuovi duplicati)
    top_contexts_combined = list(set(list(top_2_active) + list(top_2_inactive)))
    
    print(f"\n   Codici da visualizzare: {top_contexts_combined}")
    
    if len(top_contexts_combined) == 0:
        print("   ⚠️  Nessun contesto da visualizzare!")
        return
    
    try:
        fig = visualize_reverse_trace_comparison_dual(
            reconstructions_active, 
            reconstructions_inactive,
            baseline_recon_active,
            baseline_recon_inactive,
            top_contexts_combined,
            center_code_idx,
            title_prefix="Reverse Context Effect DUAL - "
        )
        
        if fig is None:
            print("   ❌ Errore: figura non creata (None)")
            return
        
        print(f"   ✅ Figura creata: {fig}")
        print(f"   ✅ Figura size: {fig.get_size_inches()}")
        
        wandb.log({"reverse_context/dual_top2_effect_traces": wandb.Image(fig)})
        print(f"   ✅ Immagine loggata su W&B come 'reverse_context/dual_top2_effect_traces'")
        plt.close(fig)
    except Exception as e:
        print(f"   ❌ Errore durante la creazione della figura: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # 3. LOG METRICHE
    # ========================================================================
    print("\n📊 Loggando metriche...")
    
    # Calcola self-context per le metriche
    self_context_mse_active = mse_diff_active[center_code_idx]
    other_contexts_mse_active = np.delete(mse_diff_active, center_code_idx)
    other_contexts_mean_active = np.mean(other_contexts_mse_active)
    
    self_context_mse_inactive = mse_diff_inactive[center_code_idx]
    other_contexts_mse_inactive = np.delete(mse_diff_inactive, center_code_idx)
    other_contexts_mean_inactive = np.mean(other_contexts_mse_inactive)
    
    # Top 2 per le metriche
    top_2_active = np.argsort(mse_diff_active)[-2:][::-1]
    top_2_inactive = np.argsort(mse_diff_inactive)[-2:][::-1]
    top_1_active = top_2_active[0]
    top_1_inactive = top_2_inactive[0]
    
    wandb.log({
        "reverse_context/center_code": center_code_idx,
        "reverse_context/center_position": center_position,
        "reverse_context/center_width": CENTER_WIDTH,
        "reverse_context/num_contexts_tested": num_codes,
        
        # Active
        "reverse_context/active/mean_mse_diff": float(np.mean(mse_diff_active)),
        "reverse_context/active/max_mse_diff": float(np.max(mse_diff_active)),
        "reverse_context/active/min_mse_diff": float(np.min(mse_diff_active)),
        "reverse_context/active/self_context_mse": float(self_context_mse_active),
        "reverse_context/active/other_contexts_mean_mse": float(other_contexts_mean_active),
        "reverse_context/active/best_context": int(top_1_active),
        "reverse_context/active/best_mse_diff": float(mse_diff_active[top_1_active]),
        "reverse_context/active/top2_contexts": [int(x) for x in top_2_active],
        
        # Inactive
        "reverse_context/inactive/mean_mse_diff": float(np.mean(mse_diff_inactive)),
        "reverse_context/inactive/max_mse_diff": float(np.max(mse_diff_inactive)),
        "reverse_context/inactive/min_mse_diff": float(np.min(mse_diff_inactive)),
        "reverse_context/inactive/self_context_mse": float(self_context_mse_inactive),
        "reverse_context/inactive/other_contexts_mean_mse": float(other_contexts_mean_inactive),
        "reverse_context/inactive/best_context": int(top_1_inactive),
        "reverse_context/inactive/best_mse_diff": float(mse_diff_inactive[top_1_inactive]),
        "reverse_context/inactive/top2_contexts": [int(x) for x in top_2_inactive],
        
        # Comparison
        "reverse_context/comparison/active_vs_inactive_mean_ratio": float(np.mean(mse_diff_active) / np.mean(mse_diff_inactive)) if np.mean(mse_diff_inactive) > 0 else 0.0,
    })
    
    print("\n✅ Analisi reverse context effect DUAL completata!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🔬 REVERSE CONTEXT PROBING DUAL: Singolo Neurone (Tracce)")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings totali: {MODEL_CONFIG['num_embeddings']}")
    print(f"   CENTER code (FIXED): {CENTER_CODE_INDEX}")
    print(f"   Center positions: {CENTER_POSITION} to {CENTER_POSITION + CENTER_WIDTH - 1}")
    print(f"   Center width: {CENTER_WIDTH} consecutive codes")
    print(f"   W&B Project: {WANDB_PROJECT}")
    print(f"\n🎯 Esperimento:")
    print(f"   Testa come CONTESTI diversi influenzano {CENTER_WIDTH} codici {CENTER_CODE_INDEX} consecutivi")
    print(f"   Sequenze: [X, X, X, X, X, 7, 7, 7, 7, 7, X, X, X, X, X]")
    
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
            "experiment_type": "reverse_context_multiple_dual",
            **MODEL_CONFIG
        },
        tags=["reverse-context-probing", "single-neuron", "dual-codebook", "traces", "multiple-center", "code-7-center"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # Carica modello
        model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # Esegui reverse context probing
        (reconstructions_active, reconstructions_inactive,
         baseline_recon_active, baseline_recon_inactive) = probe_reverse_context_effect_dual(
            model, 
            CENTER_CODE_INDEX, 
            CENTER_POSITION,
            CENTER_WIDTH,
            TEMPORAL_LENGTH, 
            device
        )
        
        # Analizza e logga risultati
        log_reverse_context_analysis_dual(
            reconstructions_active, 
            reconstructions_inactive,
            baseline_recon_active, 
            baseline_recon_inactive,
            CENTER_CODE_INDEX, 
            CENTER_POSITION
        )
        
        print("\n" + "="*70)
        print("✅ REVERSE CONTEXT PROBING DUAL COMPLETATO!")
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

