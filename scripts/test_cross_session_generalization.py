#!/usr/bin/env python3
"""
Script per testare il modello allenato su SESSIONI COMPLETAMENTE NUOVE
per verificare la capacità di generalizzazione cross-subject.

LOG SU WANDB: Tutte le metriche e visualizzazioni vengono salvate su W&B

Uso:
    python scripts/test_cross_session_generalization.py

Output:
    - Performance del modello su sessioni mai viste durante training
    - Confronto con la sessione originale di training
    - Visualizzazioni delle ricostruzioni su nuove sessioni
    - Dashboard W&B con tutti i risultati
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
from torch.utils.data import DataLoader
import wandb

from models.vqvae import CalciumVQVAE
from datasets.calcium import SimpleAllenBrainDataset

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-cross-session"
WANDB_RUN_NAME = "cross-session-test-v1"  # Cambia se vuoi run diverse

CHECKPOINT_PATH = "results/best_perfect_reconstruction.pth"
SAVE_DIR = "cross_session_analysis"

MODEL_CONFIG = {
    'num_neurons': 30,
    'num_hiddens': 512,        
    'num_residual_layers': 6,  
    'num_residual_hiddens': 256,
    'num_embeddings': 2048,
    'embedding_dim': 256,
    'commitment_cost': 0.05,
    'dropout_rate': 0.0
}

# Sessioni da testare
TRAINING_SESSION = 501474098  # Sessione usata per training

# Sessioni nuove da testare (stesso targeted_structure: VISp)
TEST_SESSIONS = [
    501559087,  # VISp, Slc17a7-IRES2-Cre
    501498760,  # VISp, provata e funzionante
    501836392,  # VISp, provata e funzionante
]

BATCH_SIZE = 32
NUM_BATCHES_TO_ANALYZE = 2  # Analizza 2 batch per sessione

# ============================================================================
# FUNZIONI HELPER
# ============================================================================

def load_trained_model(checkpoint_path, config, device):
    """Carica il modello dai pesi salvati"""
    print(f"🔄 Caricando modello da: {checkpoint_path}")
    
    model = CalciumVQVAE(
        num_neurons=config['num_neurons'],
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        dropout_rate=config.get('dropout_rate', 0.0)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Pesi caricati (epoca {checkpoint.get('epoch', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Pesi caricati")
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_session(model, session_id, device, num_batches=2):
    """Valuta il modello su una specifica sessione e logga su W&B"""
    
    print(f"\n{'='*70}")
    print(f"📊 Testando sessione: {session_id}")
    print(f"{'='*70}")
    
    try:
        # Carica dataset per questa sessione
        dataset = SimpleAllenBrainDataset(
            window_size=50,
            stride=10,
            min_neurons=30,
            session_id=session_id
        )
        
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"✅ Dataset caricato: {len(dataset)} campioni")
        
        # Raccogli metriche
        all_mse = []
        all_mae = []
        all_corr = []
        all_originals = []
        all_reconstructions = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= num_batches:
                    break
                
                neural_data = batch.to(device)
                
                # Forward pass
                vq_loss, neural_recon, perplexity, _, _ = model(neural_data)
                
                # Converti in numpy
                original = neural_data.cpu().numpy()
                reconstructed = neural_recon.cpu().numpy()
                
                # Salva per visualizzazione (primi 5 campioni del primo batch)
                if batch_idx == 0:
                    all_originals = original[:5]
                    all_reconstructions = reconstructed[:5]
                
                # Calcola metriche per ogni campione
                for i in range(original.shape[0]):
                    sample_orig = original[i].flatten()
                    sample_recon = reconstructed[i].flatten()
                    
                    mse = np.mean((sample_orig - sample_recon) ** 2)
                    mae = np.mean(np.abs(sample_orig - sample_recon))
                    
                    if np.std(sample_orig) > 1e-8 and np.std(sample_recon) > 1e-8:
                        corr, _ = pearsonr(sample_orig, sample_recon)
                        corr = corr if np.isfinite(corr) else 0
                    else:
                        corr = 0
                    
                    all_mse.append(mse)
                    all_mae.append(mae)
                    all_corr.append(corr)
        
        # Statistiche
        results = {
            'session_id': session_id,
            'num_samples': len(all_mse),
            'mse_mean': np.mean(all_mse),
            'mse_std': np.std(all_mse),
            'mae_mean': np.mean(all_mae),
            'mae_std': np.std(all_mae),
            'corr_mean': np.mean(all_corr),
            'corr_std': np.std(all_corr),
            'mse_min': np.min(all_mse),
            'mse_max': np.max(all_mse),
            'corr_min': np.min(all_corr),
            'corr_max': np.max(all_corr),
            'all_mse': all_mse,
            'all_corr': all_corr,
            'originals': all_originals,
            'reconstructions': all_reconstructions,
            'success': True
        }
        
        # Stampa risultati
        print(f"\n📈 RISULTATI:")
        print(f"   Campioni analizzati: {results['num_samples']}")
        print(f"   MSE:  {results['mse_mean']:.4f} ± {results['mse_std']:.4f}")
        print(f"   MAE:  {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
        print(f"   Corr: {results['corr_mean']:.3f} ± {results['corr_std']:.3f}")
        print(f"   Range MSE:  [{results['mse_min']:.4f}, {results['mse_max']:.4f}]")
        print(f"   Range Corr: [{results['corr_min']:.3f}, {results['corr_max']:.3f}]")
        
        # 🎯 LOG SU W&B - Metriche per sessione
        session_label = "training" if session_id == TRAINING_SESSION else f"new_{session_id}"
        
        wandb.log({
            f'{session_label}/mse_mean': results['mse_mean'],
            f'{session_label}/mse_std': results['mse_std'],
            f'{session_label}/mae_mean': results['mae_mean'],
            f'{session_label}/corr_mean': results['corr_mean'],
            f'{session_label}/corr_std': results['corr_std'],
            f'{session_label}/num_samples': results['num_samples'],
        })
        
        # 🎯 LOG SU W&B - Distribuzioni
        wandb.log({
            f'{session_label}/mse_distribution': wandb.Histogram(all_mse),
            f'{session_label}/correlation_distribution': wandb.Histogram(all_corr),
        })
        
        # 🎯 LOG SU W&B - Esempi di ricostruzione
        log_reconstruction_examples_wandb(
            all_originals, all_reconstructions, session_id, session_label
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Errore durante il test della sessione {session_id}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'session_id': session_id,
            'success': False,
            'error': str(e)
        }


def log_reconstruction_examples_wandb(originals, reconstructions, session_id, session_label):
    """Crea e logga esempi di ricostruzione su W&B"""
    
    n_samples = min(5, len(originals))
    
    for sample_idx in range(n_samples):
        # Crea figura per questo campione
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Session {session_id} - Sample {sample_idx}', fontsize=12)
        
        # Seleziona neuroni più attivi
        neuron_vars = np.var(originals[sample_idx], axis=1)
        top_neurons = np.argsort(neuron_vars)[-15:]
        
        # Original
        im1 = axes[0].imshow(originals[sample_idx, top_neurons, :], 
                            aspect='auto', cmap='viridis', vmin=-2, vmax=2)
        axes[0].set_title('Original')
        axes[0].set_ylabel('Neurons')
        axes[0].set_xlabel('Time')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Reconstruction
        im2 = axes[1].imshow(reconstructions[sample_idx, top_neurons, :], 
                            aspect='auto', cmap='viridis', vmin=-2, vmax=2)
        axes[1].set_title('Reconstruction')
        axes[1].set_xlabel('Time')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # Error
        error = np.abs(originals[sample_idx, top_neurons, :] - 
                      reconstructions[sample_idx, top_neurons, :])
        im3 = axes[2].imshow(error, aspect='auto', cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel('Time')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        
        # Log su W&B
        wandb.log({
            f'{session_label}/reconstruction_sample_{sample_idx}': wandb.Image(fig)
        })
        
        plt.close(fig)


def visualize_cross_session_results(results_dict, save_dir):
    """Crea visualizzazioni comparative tra sessioni e logga su W&B"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Filtra solo risultati successful
    successful_results = {k: v for k, v in results_dict.items() if v['success']}
    
    if len(successful_results) < 2:
        print("⚠️ Non abbastanza sessioni testate con successo per il confronto")
        return
    
    # ========================================================================
    # FIGURA 1: Confronto metriche tra sessioni
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🎯 CROSS-SESSION GENERALIZATION: Confronto Performance', 
                 fontsize=16, fontweight='bold')
    
    session_names = []
    mse_means = []
    mse_stds = []
    corr_means = []
    corr_stds = []
    
    for session_id, results in successful_results.items():
        label = "Training" if session_id == TRAINING_SESSION else f"New {session_id}"
        session_names.append(label)
        mse_means.append(results['mse_mean'])
        mse_stds.append(results['mse_std'])
        corr_means.append(results['corr_mean'])
        corr_stds.append(results['corr_std'])
    
    # Bar plot MSE
    x_pos = np.arange(len(session_names))
    colors = ['skyblue' if 'Training' in name else 'lightcoral' for name in session_names]
    
    axes[0, 0].bar(x_pos, mse_means, yerr=mse_stds, color=colors, alpha=0.7, capsize=5)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(session_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Mean Squared Error per Session')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Bar plot Correlation
    axes[0, 1].bar(x_pos, corr_means, yerr=corr_stds, color=colors, alpha=0.7, capsize=5)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(session_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_title('Mean Correlation per Session')
    axes[0, 1].set_ylim([0.8, 1.0])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Distribution MSE
    mse_distributions = [results['all_mse'] for results in successful_results.values()]
    bp1 = axes[1, 0].boxplot(mse_distributions, labels=session_names, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    axes[1, 0].set_xticklabels(session_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('MSE Distribution per Session')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Distribution Correlation
    corr_distributions = [results['all_corr'] for results in successful_results.values()]
    bp2 = axes[1, 1].boxplot(corr_distributions, labels=session_names, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    axes[1, 1].set_xticklabels(session_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].set_title('Correlation Distribution per Session')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Aggiungi statistiche
    stats_text = "GENERALIZATION SUMMARY:\n\n"
    
    if TRAINING_SESSION in successful_results:
        train_mse = successful_results[TRAINING_SESSION]['mse_mean']
        train_corr = successful_results[TRAINING_SESSION]['corr_mean']
        stats_text += f"Training Session:\n"
        stats_text += f"  MSE:  {train_mse:.4f}\n"
        stats_text += f"  Corr: {train_corr:.3f}\n\n"
    
    new_sessions_mse = [v['mse_mean'] for k, v in successful_results.items() if k != TRAINING_SESSION]
    new_sessions_corr = [v['corr_mean'] for k, v in successful_results.items() if k != TRAINING_SESSION]
    
    if new_sessions_mse:
        stats_text += f"New Sessions (avg):\n"
        stats_text += f"  MSE:  {np.mean(new_sessions_mse):.4f}\n"
        stats_text += f"  Corr: {np.mean(new_sessions_corr):.3f}\n\n"
        
        if TRAINING_SESSION in successful_results:
            degradation_mse = (np.mean(new_sessions_mse) - train_mse) / train_mse * 100
            degradation_corr = (train_corr - np.mean(new_sessions_corr)) / train_corr * 100
            stats_text += f"Performance Change:\n"
            stats_text += f"  MSE: {degradation_mse:+.1f}%\n"
            stats_text += f"  Corr: {degradation_corr:+.1f}%"
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Salva localmente
    plt.savefig(save_dir / 'cross_session_comparison.png', dpi=200, bbox_inches='tight')
    
    # 🎯 LOG SU W&B
    wandb.log({
        "cross_session/comparison_plot": wandb.Image(fig)
    })
    
    plt.close()
    
    print(f"✅ Visualizzazione salvata localmente e su W&B")
    
    # ========================================================================
    # FIGURA 2: Esempi di ricostruzione per ogni sessione
    # ========================================================================
    n_sessions = len(successful_results)
    n_samples = 3
    
    fig, axes = plt.subplots(n_sessions, n_samples * 3, figsize=(15, 4*n_sessions))
    fig.suptitle('🔬 Reconstruction Examples Across Sessions', fontsize=14, fontweight='bold')
    
    if n_sessions == 1:
        axes = axes.reshape(1, -1)
    
    for session_idx, (session_id, results) in enumerate(successful_results.items()):
        if not results['success'] or 'originals' not in results:
            continue
        
        originals = results['originals']
        reconstructions = results['reconstructions']
        
        session_label = "Training" if session_id == TRAINING_SESSION else f"New {session_id}"
        
        for sample_idx in range(min(n_samples, len(originals))):
            neuron_vars = np.var(originals[sample_idx], axis=1)
            top_neurons = np.argsort(neuron_vars)[-15:]
            
            col_base = sample_idx * 3
            
            # Original
            ax_orig = axes[session_idx, col_base]
            ax_orig.imshow(originals[sample_idx, top_neurons, :], 
                          aspect='auto', cmap='viridis', vmin=-2, vmax=2)
            if sample_idx == 0:
                ax_orig.set_ylabel(f'{session_label}\nNeurons', fontsize=10)
            if session_idx == 0:
                ax_orig.set_title('Original', fontsize=10)
            
            # Reconstruction
            ax_recon = axes[session_idx, col_base + 1]
            ax_recon.imshow(reconstructions[sample_idx, top_neurons, :], 
                           aspect='auto', cmap='viridis', vmin=-2, vmax=2)
            if session_idx == 0:
                ax_recon.set_title('Reconstruction', fontsize=10)
            
            # Error
            error = np.abs(originals[sample_idx, top_neurons, :] - 
                          reconstructions[sample_idx, top_neurons, :])
            ax_error = axes[session_idx, col_base + 2]
            ax_error.imshow(error, aspect='auto', cmap='Reds', vmin=0, vmax=1)
            if session_idx == 0:
                ax_error.set_title('Absolute Error', fontsize=10)
            
            for ax in [ax_orig, ax_recon, ax_error]:
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.tight_layout()
    
    # Salva localmente
    plt.savefig(save_dir / 'cross_session_examples.png', dpi=200, bbox_inches='tight')
    
    # 🎯 LOG SU W&B
    wandb.log({
        "cross_session/examples_grid": wandb.Image(fig)
    })
    
    plt.close()
    
    #print(f"✅ Esempi salvati localmente e su W&B")_distributions = [results['all_corr'] for results in successful_results.values()]
    bp2 = axes[1, 1].boxplot(corr_distributions, labels=session_names, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    axes[1, 1].set_xticklabels(session_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].set_title('Correlation Distribution per Session')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Aggiungi statistiche
    stats_text = "GENERALIZATION SUMMARY:\n\n"
    
    if TRAINING_SESSION in successful_results:
        train_mse = successful_results[TRAINING_SESSION]['mse_mean']
        train_corr = successful_results[TRAINING_SESSION]['corr_mean']
        stats_text += f"Training Session:\n"
        stats_text += f"  MSE:  {train_mse:.4f}\n"
        stats_text += f"  Corr: {train_corr:.3f}\n\n"
    
    new_sessions_mse = [v['mse_mean'] for k, v in successful_results.items() if k != TRAINING_SESSION]
    new_sessions_corr = [v['corr_mean'] for k, v in successful_results.items() if k != TRAINING_SESSION]
    
    if new_sessions_mse:
        stats_text += f"New Sessions (avg):\n"
        stats_text += f"  MSE:  {np.mean(new_sessions_mse):.4f}\n"
        stats_text += f"  Corr: {np.mean(new_sessions_corr):.3f}\n\n"
        
        if TRAINING_SESSION in successful_results:
            degradation_mse = (np.mean(new_sessions_mse) - train_mse) / train_mse * 100
            degradation_corr = (train_corr - np.mean(new_sessions_corr)) / train_corr * 100
            stats_text += f"Performance Change:\n"
            stats_text += f"  MSE: {degradation_mse:+.1f}%\n"
            stats_text += f"  Corr: {degradation_corr:+.1f}%"
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'cross_session_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizzazione salvata: {save_dir / 'cross_session_comparison.png'}")
    
    # ========================================================================
    # FIGURA 2: Esempi di ricostruzione per ogni sessione
    # ========================================================================
    n_sessions = len(successful_results)
    n_samples = 3  # Mostra 3 campioni per sessione
    
    fig, axes = plt.subplots(n_sessions, n_samples * 3, figsize=(15, 4*n_sessions))
    fig.suptitle('🔬 Reconstruction Examples Across Sessions', fontsize=14, fontweight='bold')
    
    if n_sessions == 1:
        axes = axes.reshape(1, -1)
    
    for session_idx, (session_id, results) in enumerate(successful_results.items()):
        if not results['success'] or 'originals' not in results:
            continue
        
        originals = results['originals']
        reconstructions = results['reconstructions']
        
        session_label = "Training" if session_id == TRAINING_SESSION else f"New {session_id}"
        
        for sample_idx in range(min(n_samples, len(originals))):
            # Seleziona neuroni più attivi
            neuron_vars = np.var(originals[sample_idx], axis=1)
            top_neurons = np.argsort(neuron_vars)[-15:]
            
            col_base = sample_idx * 3
            
            # Original
            ax_orig = axes[session_idx, col_base]
            ax_orig.imshow(originals[sample_idx, top_neurons, :], 
                          aspect='auto', cmap='viridis', vmin=-2, vmax=2)
            if sample_idx == 0:
                ax_orig.set_ylabel(f'{session_label}\nNeurons', fontsize=10)
            if session_idx == 0:
                ax_orig.set_title('Original', fontsize=10)
            
            # Reconstruction
            ax_recon = axes[session_idx, col_base + 1]
            ax_recon.imshow(reconstructions[sample_idx, top_neurons, :], 
                           aspect='auto', cmap='viridis', vmin=-2, vmax=2)
            if session_idx == 0:
                ax_recon.set_title('Reconstruction', fontsize=10)
            
            # Error
            error = np.abs(originals[sample_idx, top_neurons, :] - 
                          reconstructions[sample_idx, top_neurons, :])
            ax_error = axes[session_idx, col_base + 2]
            ax_error.imshow(error, aspect='auto', cmap='Reds', vmin=0, vmax=1)
            if session_idx == 0:
                ax_error.set_title('Absolute Error', fontsize=10)
            
            # Rimuovi tick labels
            for ax in [ax_orig, ax_recon, ax_error]:
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'cross_session_examples.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Esempi salvati: {save_dir / 'cross_session_examples.png'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("🎯 CROSS-SESSION GENERALIZATION TEST")
    print("="*70)
    print(f"📁 Modello: {CHECKPOINT_PATH}")
    print(f"🧪 Sessione training: {TRAINING_SESSION}")
    print(f"🔬 Sessioni test: {TEST_SESSIONS}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 🎯 INIZIALIZZA W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "checkpoint_path": CHECKPOINT_PATH,
            "training_session": TRAINING_SESSION,
            "test_sessions": TEST_SESSIONS,
            "batch_size": BATCH_SIZE,
            "num_batches_analyzed": NUM_BATCHES_TO_ANALYZE,
            **MODEL_CONFIG
        },
        tags=["cross-session", "generalization", "evaluation"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    # Carica modello
    model = load_trained_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
    
    # Dizionario per salvare risultati
    all_results = {}
    
    # Test su sessione di training (come baseline)
    print("\n" + "="*70)
    print("📊 BASELINE: Testando su sessione di TRAINING")
    print("="*70)
    results_training = evaluate_session(model, TRAINING_SESSION, device, NUM_BATCHES_TO_ANALYZE)
    all_results[TRAINING_SESSION] = results_training
    
    # Test su sessioni nuove
    print("\n" + "="*70)
    print("🔬 CROSS-SESSION: Testando su sessioni NUOVE")
    print("="*70)
    
    for session_id in TEST_SESSIONS:
        results = evaluate_session(model, session_id, device, NUM_BATCHES_TO_ANALYZE)
        all_results[session_id] = results
    
    # Crea visualizzazioni comparative
    print("\n" + "="*70)
    print("🎨 Creando visualizzazioni comparative...")
    print("="*70)
    visualize_cross_session_results(all_results, save_dir)
    
    # Summary finale
    print("\n" + "="*70)
    print("📊 SUMMARY FINALE")
    print("="*70)
    
    successful_count = sum(1 for r in all_results.values() if r['success'])
    print(f"✅ Sessioni testate con successo: {successful_count}/{len(all_results)}")
    
    # 🎯 LOG SUMMARY SU W&B
    summary_metrics = {}
    
    if TRAINING_SESSION in all_results and all_results[TRAINING_SESSION]['success']:
        train_results = all_results[TRAINING_SESSION]
        print(f"\n📈 TRAINING SESSION ({TRAINING_SESSION}):")
        print(f"   MSE:  {train_results['mse_mean']:.4f} ± {train_results['mse_std']:.4f}")
        print(f"   Corr: {train_results['corr_mean']:.3f} ± {train_results['corr_std']:.3f}")
        
        summary_metrics['training_mse'] = train_results['mse_mean']
        summary_metrics['training_corr'] = train_results['corr_mean']
    
    new_session_results = [v for k, v in all_results.items() 
                          if k != TRAINING_SESSION and v['success']]
    
    if new_session_results:
        avg_mse = np.mean([r['mse_mean'] for r in new_session_results])
        avg_corr = np.mean([r['corr_mean'] for r in new_session_results])
        
        print(f"\n🔬 NEW SESSIONS (Average):")
        print(f"   MSE:  {avg_mse:.4f}")
        print(f"   Corr: {avg_corr:.3f}")
        
        summary_metrics['new_sessions_mse_avg'] = avg_mse
        summary_metrics['new_sessions_corr_avg'] = avg_corr
        
        if TRAINING_SESSION in all_results and all_results[TRAINING_SESSION]['success']:
            train_mse = all_results[TRAINING_SESSION]['mse_mean']
            train_corr = all_results[TRAINING_SESSION]['corr_mean']
            
            change_mse = (avg_mse - train_mse) / train_mse * 100
            change_corr = (train_corr - avg_corr) / train_corr * 100
            
            print(f"\n💡 GENERALIZZAZIONE:")
            print(f"   Cambio MSE:  {change_mse:+.1f}%")
            print(f"   Cambio Corr: {change_corr:+.1f}%")
            
            summary_metrics['generalization_mse_change_pct'] = change_mse
            summary_metrics['generalization_corr_change_pct'] = change_corr
            
            if abs(change_mse) < 50 and abs(change_corr) < 10:
                generalization_quality = "Excellent"
                print(f"   ✅ Buona generalizzazione cross-session!")
            elif abs(change_mse) < 100:
                generalization_quality = "Moderate"
                print(f"   ⚠️  Generalizzazione moderata")
            else:
                generalization_quality = "Poor"
                print(f"   ❌ Generalizzazione scarsa")
            
            summary_metrics['generalization_quality'] = generalization_quality
    
    # 🎯 LOG SUMMARY TABLE SU W&B
    summary_table_data = []
    for session_id, results in all_results.items():
        if results['success']:
            session_type = "Training" if session_id == TRAINING_SESSION else "New"
            summary_table_data.append([
                session_id,
                session_type,
                results['num_samples'],
                results['mse_mean'],
                results['mse_std'],
                results['corr_mean'],
                results['corr_std']
            ])
    
    summary_table = wandb.Table(
        columns=["Session ID", "Type", "Samples", "MSE Mean", "MSE Std", "Corr Mean", "Corr Std"],
        data=summary_table_data
    )
    
    wandb.log({
        "summary/session_comparison_table": summary_table,
        **summary_metrics
    })
    
    print(f"\n📁 Risultati salvati localmente in: {save_dir}/")
    print(f"🌐 Risultati salvati su W&B: {wandb.run.url}")
    print("="*70)
    
    # 🎯 FINISH W&B
    wandb.finish()
    print("✅ W&B run completato!")


if __name__ == "__main__":
    main()