#!/usr/bin/env python3
"""
Context Probing DUAL CODEBOOK per SINGOLO NEURONE: Testa l'effetto del contesto sui due codebook.

Strategia:
- Codice "nullo" (baseline): Codice 7 per entrambi i codebook
- Crea sequenze: [7, 7, 7, X, 7, 7, 7] dove X varia
- Analizza separatamente per ACTIVE e INACTIVE codebook
- Vedi come il codice centrale X influenza la ricostruzione del singolo neurone

Uso:
    python scripts/probing_single_neuron_dual.py
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
WANDB_PROJECT = "calcium-vqvae-context-probing-single-neuron-dual"
WANDB_RUN_NAME = f"context-probe-1neuron-dual-{datetime.now().strftime('%m%d-%H%M')}"

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

# ✅ CONFIGURAZIONE ESPERIMENTO
NULL_CODE_INDEX = 7        # Codice "nullo" di base (baseline)
TEMPORAL_LENGTH = 15       # Lunghezza totale sequenza
CENTER_POSITION = 7        # Posizione del codice variabile (metà = 7)

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


def probe_context_effect_dual(model, null_code_idx, center_position, temporal_length, device):
    """
    Prova tutti i codici come "centro" in un contesto di NULL codes.
    Esegue per entrambi i codebook (active e inactive).
    
    Returns:
        reconstructions_active: Array (num_codes_per_book, 1, output_time)
        reconstructions_inactive: Array (num_codes_per_book, 1, output_time)
        baseline_recon_active: Array (1, output_time)
        baseline_recon_inactive: Array (1, output_time)
    """
    
    vq = model.vector_quantization
    n_e_per_codebook = vq.n_e_per_codebook  # Numero embeddings per codebook
    
    # Verifica che NULL_CODE_INDEX sia valido
    if null_code_idx >= n_e_per_codebook:
        raise ValueError(f"NULL_CODE_INDEX ({null_code_idx}) deve essere < n_e_per_codebook ({n_e_per_codebook})")
    
    # Ottieni embeddings dei due codebook
    embeddings_active = vq.embedding_active.weight.data
    embeddings_inactive = vq.embedding_inactive.weight.data
    
    print(f"\n🔬 Context Probing DUAL:")
    print(f"   NULL code: {null_code_idx}")
    print(f"   Center position: {center_position}/{temporal_length}")
    print(f"   Active codebook: {n_e_per_codebook} codes (indici 0-{n_e_per_codebook-1})")
    print(f"   Inactive codebook: {n_e_per_codebook} codes (indici 0-{n_e_per_codebook-1})")
    print(f"   Testing {n_e_per_codebook} different center codes per codebook...")
    
    all_reconstructions_active = []
    all_reconstructions_inactive = []
    
    model.eval()
    with torch.no_grad():
        # ====================================================================
        # ACTIVE CODEBOOK
        # ====================================================================
        print("\n   🔴 ACTIVE Codebook:")
        
        # BASELINE: Solo NULL codes
        print("      🔵 Baseline: Full NULL sequence...")
        null_embedding_active = embeddings_active[null_code_idx]
        baseline_sequence_active = null_embedding_active.unsqueeze(1).repeat(1, temporal_length).unsqueeze(0).to(device)
        baseline_recon_active = model.decode(baseline_sequence_active).cpu().numpy()[0]  # Shape: (1, output_time)
        
        # CONTEXT TEST: Varia il codice centrale
        print("      🔴 Testing center codes...")
        for center_code_idx in range(n_e_per_codebook):
            sequence = create_context_sequence(
                embeddings_active, 
                null_code_idx, 
                center_code_idx,
                temporal_length, 
                center_position, 
                device
            )
            reconstruction = model.decode(sequence)
            all_reconstructions_active.append(reconstruction.cpu().numpy()[0])  # Shape: (1, output_time)
            
            if (center_code_idx + 1) % 50 == 0:
                print(f"         Processati {center_code_idx + 1}/{n_e_per_codebook} center codes...")
        
        # ====================================================================
        # INACTIVE CODEBOOK
        # ====================================================================
        print("\n   🔵 INACTIVE Codebook:")
        
        # BASELINE: Solo NULL codes
        print("      🔵 Baseline: Full NULL sequence...")
        null_embedding_inactive = embeddings_inactive[null_code_idx]
        baseline_sequence_inactive = null_embedding_inactive.unsqueeze(1).repeat(1, temporal_length).unsqueeze(0).to(device)
        baseline_recon_inactive = model.decode(baseline_sequence_inactive).cpu().numpy()[0]  # Shape: (1, output_time)
        
        # CONTEXT TEST: Varia il codice centrale
        print("      🔴 Testing center codes...")
        for center_code_idx in range(n_e_per_codebook):
            sequence = create_context_sequence(
                embeddings_inactive, 
                null_code_idx, 
                center_code_idx,
                temporal_length, 
                center_position, 
                device
            )
            reconstruction = model.decode(sequence)
            all_reconstructions_inactive.append(reconstruction.cpu().numpy()[0])  # Shape: (1, output_time)
            
            if (center_code_idx + 1) % 50 == 0:
                print(f"         Processati {center_code_idx + 1}/{n_e_per_codebook} center codes...")
    
    reconstructions_active = np.array(all_reconstructions_active)  # Shape: (num_codes, 1, output_time)
    reconstructions_inactive = np.array(all_reconstructions_inactive)  # Shape: (num_codes, 1, output_time)
    
    print(f"\n✅ Active reconstructions shape: {reconstructions_active.shape}")
    print(f"✅ Inactive reconstructions shape: {reconstructions_inactive.shape}")
    print(f"✅ Active baseline shape: {baseline_recon_active.shape}")
    print(f"✅ Inactive baseline shape: {baseline_recon_inactive.shape}")
    
    return (reconstructions_active, reconstructions_inactive, 
            baseline_recon_active, baseline_recon_inactive)


def visualize_trace_comparison_dual(reconstructions_active, reconstructions_inactive,
                                   baseline_recon_active, baseline_recon_inactive,
                                   code_indices, null_code_idx, title_prefix=""):
    """
    Visualizza tracce temporali per confronto DUAL (invece di heatmap)
    
    Args:
        reconstructions_active: (num_codes, 1, time)
        reconstructions_inactive: (num_codes, 1, time)
        baseline_recon_active: (1, time)
        baseline_recon_inactive: (1, time)
        code_indices: lista di codici da visualizzare
        null_code_idx: indice del codice NULL
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
    axes[0, 0].plot(time_points, baseline_recon_active[0, :], 'gray', linewidth=2, 
                   label=f'BASELINE (all code {null_code_idx})', alpha=0.8)
    axes[0, 0].set_ylabel('Activity')
    axes[0, 0].set_title(f'{title_prefix}ACTIVE BASELINE: Full NULL (code {null_code_idx})', 
                        fontweight='bold', fontsize=12)
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, output_time - 1])
    
    # === INACTIVE BASELINE ===
    axes[1, 0].plot(time_points, baseline_recon_inactive[0, :], 'gray', linewidth=2, 
                   label=f'BASELINE (all code {null_code_idx})', alpha=0.8)
    axes[1, 0].set_ylabel('Activity')
    axes[1, 0].set_title(f'{title_prefix}INACTIVE BASELINE: Full NULL (code {null_code_idx})', 
                        fontweight='bold', fontsize=12)
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, output_time - 1])
    
    # === CENTER CODES ===
    for plot_idx, code_idx in enumerate(code_indices, 1):
        
        # ACTIVE
        recon_trace_active = reconstructions_active[code_idx, 0, :]
        axes[0, plot_idx].plot(time_points, baseline_recon_active[0, :], 'gray', 
                              linewidth=1, alpha=0.4, label='Baseline')
        axes[0, plot_idx].plot(time_points, recon_trace_active, 'b-', 
                              linewidth=2, label=f'CENTER=Code {code_idx}', alpha=0.8)
        
        diff_active = recon_trace_active - baseline_recon_active[0, :]
        axes[0, plot_idx].fill_between(time_points, baseline_recon_active[0, :], recon_trace_active,
                                      where=(diff_active > 0), color='green', alpha=0.2, 
                                      label='Increased activity')
        axes[0, plot_idx].fill_between(time_points, baseline_recon_active[0, :], recon_trace_active,
                                      where=(diff_active < 0), color='red', alpha=0.2, 
                                      label='Decreased activity')
        
        mse_diff_active = np.mean(diff_active ** 2)
        mae_diff_active = np.mean(np.abs(diff_active))
        if np.std(baseline_recon_active[0, :]) > 1e-8 and np.std(recon_trace_active) > 1e-8:
            corr_active, _ = pearsonr(baseline_recon_active[0, :], recon_trace_active)
        else:
            corr_active = 1.0
        
        axes[0, plot_idx].set_ylabel('Activity')
        axes[0, plot_idx].set_title(f'ACTIVE: CENTER=Code {code_idx} | '
                                    f'MSE diff={mse_diff_active:.4f}, Corr={corr_active:.3f}',
                                    fontweight='bold', fontsize=11)
        axes[0, plot_idx].legend(loc='upper right', fontsize=9)
        axes[0, plot_idx].grid(True, alpha=0.3)
        axes[0, plot_idx].set_xlim([0, output_time - 1])
        
        # INACTIVE
        recon_trace_inactive = reconstructions_inactive[code_idx, 0, :]
        axes[1, plot_idx].plot(time_points, baseline_recon_inactive[0, :], 'gray', 
                              linewidth=1, alpha=0.4, label='Baseline')
        axes[1, plot_idx].plot(time_points, recon_trace_inactive, 'b-', 
                              linewidth=2, label=f'CENTER=Code {code_idx}', alpha=0.8)
        
        diff_inactive = recon_trace_inactive - baseline_recon_inactive[0, :]
        axes[1, plot_idx].fill_between(time_points, baseline_recon_inactive[0, :], recon_trace_inactive,
                                      where=(diff_inactive > 0), color='green', alpha=0.2, 
                                      label='Increased activity')
        axes[1, plot_idx].fill_between(time_points, baseline_recon_inactive[0, :], recon_trace_inactive,
                                      where=(diff_inactive < 0), color='red', alpha=0.2, 
                                      label='Decreased activity')
        
        mse_diff_inactive = np.mean(diff_inactive ** 2)
        mae_diff_inactive = np.mean(np.abs(diff_inactive))
        if np.std(baseline_recon_inactive[0, :]) > 1e-8 and np.std(recon_trace_inactive) > 1e-8:
            corr_inactive, _ = pearsonr(baseline_recon_inactive[0, :], recon_trace_inactive)
        else:
            corr_inactive = 1.0
        
        axes[1, plot_idx].set_ylabel('Activity')
        axes[1, plot_idx].set_title(f'INACTIVE: CENTER=Code {code_idx} | '
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


def log_context_analysis_dual(reconstructions_active, reconstructions_inactive,
                             baseline_recon_active, baseline_recon_inactive,
                             null_code_idx, center_position):
    """
    Analizza e logga i risultati del context probing con TRACCE per entrambi i codebook.
    
    Args:
        reconstructions_active: (num_codes, 1, time)
        reconstructions_inactive: (num_codes, 1, time)
        baseline_recon_active: (1, time)
        baseline_recon_inactive: (1, time)
    """
    
    num_codes = reconstructions_active.shape[0]
    output_time = reconstructions_active.shape[2]
    
    print(f"\n📊 Analisi Context Effect DUAL...")
    
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
    # 2. VISUALIZZA IL MIGLIOR CODICE PER CODEBOOK (UNICA IMMAGINE)
    # ========================================================================
    print("🎨 Visualizzando il miglior codice per codebook...")
    
    # Trova il codice con maggiore effetto per ACTIVE e INACTIVE
    top_1_active = np.argmax(mse_diff_active)
    top_1_inactive = np.argmax(mse_diff_inactive)
    
    print(f"\n   Miglior ACTIVE codice:")
    print(f"      Code {top_1_active}: MSE diff={mse_diff_active[top_1_active]:.4f}, MAE diff={mae_diff_active[top_1_active]:.4f}")
    
    print(f"\n   Miglior INACTIVE codice:")
    print(f"      Code {top_1_inactive}: MSE diff={mse_diff_inactive[top_1_inactive]:.4f}, MAE diff={mae_diff_inactive[top_1_inactive]:.4f}")
    
    # Plot solo i migliori codici (uno per codebook)
    top_codes = [top_1_active, top_1_inactive]
    # Rimuovi duplicati mantenendo l'ordine
    top_codes_unique = []
    for code in top_codes:
        if code not in top_codes_unique:
            top_codes_unique.append(code)
    
    print(f"\n   Codici da visualizzare: {top_codes_unique}")
    
    if len(top_codes_unique) == 0:
        print("   ⚠️  Nessun codice da visualizzare!")
        return
    
    try:
        fig = visualize_trace_comparison_dual(
            reconstructions_active, 
            reconstructions_inactive,
            baseline_recon_active,
            baseline_recon_inactive,
            top_codes_unique,
            null_code_idx,
            title_prefix="Context Effect DUAL - "
        )
        
        if fig is None:
            print("   ❌ Errore: figura non creata (None)")
            return
        
        print(f"   ✅ Figura creata: {fig}")
        print(f"   ✅ Figura size: {fig.get_size_inches()}")
        
        wandb.log({"context/dual_best_effect_traces": wandb.Image(fig)})
        print(f"   ✅ Immagine loggata su W&B come 'context/dual_best_effect_traces'")
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
    
    wandb.log({
        "context/null_code": null_code_idx,
        "context/center_position": center_position,
        "context/num_codes_tested": num_codes,
        
        # Active
        "context/active/mean_mse_diff": float(np.mean(mse_diff_active)),
        "context/active/max_mse_diff": float(np.max(mse_diff_active)),
        "context/active/min_mse_diff": float(np.min(mse_diff_inactive)),
        "context/active/best_code": int(top_1_active),
        "context/active/best_mse_diff": float(mse_diff_active[top_1_active]),
        
        # Inactive
        "context/inactive/mean_mse_diff": float(np.mean(mse_diff_inactive)),
        "context/inactive/max_mse_diff": float(np.max(mse_diff_inactive)),
        "context/inactive/min_mse_diff": float(np.min(mse_diff_inactive)),
        "context/inactive/best_code": int(top_1_inactive),
        "context/inactive/best_mse_diff": float(mse_diff_inactive[top_1_inactive]),
        
        # Comparison
        "context/comparison/active_vs_inactive_mean_ratio": float(np.mean(mse_diff_active) / np.mean(mse_diff_inactive)) if np.mean(mse_diff_inactive) > 0 else 0.0,
    })
    
    print("\n✅ Analisi context effect DUAL completata!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🔬 CONTEXT PROBING DUAL: Singolo Neurone (Tracce Temporali)")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings totali: {MODEL_CONFIG['num_embeddings']}")
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
        tags=["context-probing", "single-neuron", "dual-codebook", "traces", "code-interaction"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # Carica modello
        model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # Esegui context probing
        (reconstructions_active, reconstructions_inactive,
         baseline_recon_active, baseline_recon_inactive) = probe_context_effect_dual(
            model, 
            NULL_CODE_INDEX, 
            CENTER_POSITION, 
            TEMPORAL_LENGTH, 
            device
        )
        
        # Analizza e logga risultati
        log_context_analysis_dual(
            reconstructions_active, 
            reconstructions_inactive,
            baseline_recon_active, 
            baseline_recon_inactive,
            NULL_CODE_INDEX, 
            CENTER_POSITION
        )
        
        print("\n" + "="*70)
        print("✅ CONTEXT PROBING DUAL COMPLETATO!")
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

