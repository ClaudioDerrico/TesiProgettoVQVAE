#!/usr/bin/env python3
"""
Analisi Correlazione vs Distanza - DUAL Codebook CONSTRAINED Version

✨ CARATTERISTICHE:
- Calcola correlazioni temporali e spettrali tra codice base e tutti gli altri
- Calcola distanze cosine ed euclidean
- Gestisce separatamente active e inactive codebooks
- Salva tutto in JSON locali e W&B
- Gestione memoria ottimizzata per evitare crash

Output JSON contiene per ogni coppia:
- code_pair: [base_code, other_code]
- cosine_distance: float
- euclidean_distance: float
- temporal_correlation: float
- spectral_correlation: float
- base_codebook_type: "active" o "inactive"
- target_codebook_type: "active" o "inactive"

Uso:
    python scripts/codebook_corr_distance_metriche_duale_constrained.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr

from models.dual_vqvae import DualCalciumVQVAE

# ============================================================================
# ⚙️ CONFIGURAZIONE - MODIFICA QUI PER CAMBIARE MODELLO
# ============================================================================

# ✅ MODIFICA QUI: Path al checkpoint constrained
CHECKPOINT_PATH = "results/dual_codebook_constrained/best_dual_model_32_constrained.pth"

# Directory output
OUTPUT_DIR = "correlazione_distanze_temporali_spettrali_dual_constrained"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-correlation-distance-dual-constrained"
WANDB_RUN_NAME = f"corr-dist-dual-constrained-{datetime.now().strftime('%m%d-%H%M')}"

# ⚙️ CONFIGURAZIONE MODELLO: Deve corrispondere al checkpoint
MODEL_CONFIG = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 32,         # ⬅️ CAMBIA QUI: 16, 32, 64, 128, 256, 512, 1024, etc.
    'embedding_dim': 32,          # ⬅️ CAMBIA QUI: tipicamente = num_embeddings per piccoli, 64 per grandi
    'commitment_cost': 0.25,
    'dropout_rate': 0.1,
    'use_quantizer': True,
    'threshold': 0.0,
    'threshold_type': 'adaptive',
}

# Parametri
TEMPORAL_LENGTH = 15
BATCH_SIZE = 4  # Batch size ridotto per evitare out of memory

# ✨ Automaticamente analizza TUTTI i codici come base
# ⬅️ CAMBIA QUI: Deve corrispondere a num_embeddings
ALL_BASE_CODES = list(range(32))  # [0, 1, 2, ..., 31] per 32 embeddings

# ============================================================================
# FUNZIONI - CARICAMENTO MODELLO DUAL
# ============================================================================

def load_dual_model(checkpoint_path, config, device):
    """Carica il modello dual allenato con gestione memoria ottimizzata"""
    print(f"🔄 Caricando modello DUAL CONSTRAINED da: {checkpoint_path}")
    
    # ✅ Carica checkpoint su CPU per evitare out of memory
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Auto-infer num_embeddings se necessario
    if 'model_config' in checkpoint:
        model_config_checkpoint = checkpoint['model_config']
        if 'num_embeddings' in model_config_checkpoint:
            config['num_embeddings'] = model_config_checkpoint['num_embeddings']
            # Aggiorna anche ALL_BASE_CODES
            global ALL_BASE_CODES
            ALL_BASE_CODES = list(range(config['num_embeddings']))
            print(f"   ✅ Auto-rilevato num_embeddings: {config['num_embeddings']}")
    
    # Crea modello su CPU
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
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Modello DUAL CONSTRAINED caricato (epoca {checkpoint.get('epoch', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Modello DUAL CONSTRAINED caricato")
    
    # Sposta su GPU solo se disponibile e se device è cuda
    if device.type == 'cuda':
        try:
            model = model.to(device)
            print(f"✅ Modello spostato su {device}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  GPU out of memory, usando CPU invece")
                device = torch.device('cpu')
                model = model.to(device)
            else:
                raise e
    else:
        model = model.to(device)
    
    model.eval()
    
    return model, device


def extract_dual_codebook_embeddings(model):
    """Estrae gli embeddings di entrambi i codebook"""
    vq = model.vector_quantization
    n_e_per_codebook = vq.n_e_per_codebook
    
    # Estrai embeddings
    active_emb = vq.embedding_active.weight.data.cpu().numpy()
    inactive_emb = vq.embedding_inactive.weight.data.cpu().numpy()
    
    # Combina per analisi globale
    all_embeddings = np.vstack([active_emb, inactive_emb])
    split_idx = n_e_per_codebook
    
    print(f"✅ Codebook embeddings estratti:")
    print(f"   Active: {active_emb.shape}")
    print(f"   Inactive: {inactive_emb.shape}")
    print(f"   Total: {all_embeddings.shape}")
    
    return active_emb, inactive_emb, all_embeddings, split_idx


def get_codebook_type(code_idx, split_idx):
    """Determina se un codice appartiene a active o inactive"""
    return "active" if code_idx < split_idx else "inactive"

# ============================================================================
# FUNZIONI - DECODIFICA
# ============================================================================

def decode_code_dual(model, code_idx, codebook_type, device, temporal_length):
    """Decodifica un singolo codice dal codebook specificato"""
    vq = model.vector_quantization
    
    if codebook_type == "active":
        codebook_idx = code_idx
        embedding = vq.embedding_active.weight[codebook_idx]
    else:
        codebook_idx = code_idx - vq.n_e_per_codebook
        embedding = vq.embedding_inactive.weight[codebook_idx]
    
    sequence = embedding.unsqueeze(1).repeat(1, temporal_length)
    sequence = sequence.unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstruction = model.decode(sequence)
    
    trace = reconstruction.cpu().numpy()[0, 0, :]
    return trace


def decode_codes_batch_dual(model, code_indices, split_idx, device, temporal_length, batch_size=4):
    """Decodifica multipli codici in batch (gestisce active/inactive)"""
    all_traces = []
    num_codes = len(code_indices)
    
    vq = model.vector_quantization
    model.eval()
    
    with torch.no_grad():
        for i in range(0, num_codes, batch_size):
            batch_indices = code_indices[i:i+batch_size]
            batch_embeddings = []
            
            for code_idx in batch_indices:
                if code_idx < split_idx:
                    # Active codebook
                    codebook_idx = code_idx
                    embedding = vq.embedding_active.weight[codebook_idx]
                else:
                    # Inactive codebook
                    codebook_idx = code_idx - vq.n_e_per_codebook
                    embedding = vq.embedding_inactive.weight[codebook_idx]
                
                batch_embeddings.append(embedding)
            
            batch_embeddings = torch.stack(batch_embeddings)
            batch_sequences = batch_embeddings.unsqueeze(2).repeat(1, 1, temporal_length)
            batch_sequences = batch_sequences.to(device)
            
            reconstructions = model.decode(batch_sequences)
            all_traces.append(reconstructions.cpu().numpy()[:, 0, :])
            
            # Pulisci cache GPU dopo ogni batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return np.concatenate(all_traces, axis=0)

# ============================================================================
# FUNZIONI - ANALISI SPETTRALE
# ============================================================================

def compute_fft_spectrum(trace, sample_rate=1.0):
    """Calcola lo spettro di frequenza usando FFT"""
    n = len(trace)
    trace_centered = trace - np.mean(trace)
    fft_vals = np.fft.fft(trace_centered)
    freqs = np.fft.fftfreq(n, d=1.0/sample_rate)
    
    positive_freq_idx = freqs >= 0
    freqs_positive = freqs[positive_freq_idx]
    fft_positive = fft_vals[positive_freq_idx]
    power = np.abs(fft_positive) ** 2
    
    return freqs_positive, power

# ============================================================================
# FUNZIONI - CORRELAZIONI
# ============================================================================

def compute_temporal_correlation(trace_a, trace_b):
    """Calcola correlazione temporale con handling robusto"""
    std_a = np.std(trace_a)
    std_b = np.std(trace_b)
    
    if std_a < 1e-8 or std_b < 1e-8:
        if np.allclose(trace_a, trace_b, atol=1e-8):
            return 1.0
        elif std_a < 1e-8 and std_b < 1e-8:
            return 0.0
        else:
            return np.nan
    
    try:
        corr, _ = pearsonr(trace_a, trace_b)
        return corr if not np.isnan(corr) else np.nan
    except:
        return np.nan


def compute_spectral_correlation(trace_a, trace_b):
    """Calcola correlazione spettrale con handling robusto"""
    freqs_a, power_a = compute_fft_spectrum(trace_a)
    freqs_b, power_b = compute_fft_spectrum(trace_b)
    
    power_a_norm = power_a / (np.sum(power_a) + 1e-8)
    power_b_norm = power_b / (np.sum(power_b) + 1e-8)
    
    if len(power_a_norm) > 1 and np.std(power_a_norm) > 1e-8 and np.std(power_b_norm) > 1e-8:
        try:
            corr, _ = pearsonr(power_a_norm, power_b_norm)
            return corr if not np.isnan(corr) else np.nan
        except:
            return np.nan
    return np.nan

# ============================================================================
# FUNZIONI - ANALISI PRINCIPALE
# ============================================================================

def analyze_dual_code_pairs(model, all_embeddings, split_idx, device, temporal_length, base_code_idx, batch_size=4):
    """
    Analizza tutte le coppie di codici (base vs others) per dual codebook
    
    Returns:
        List of dicts, ognuno contenente:
        - code_pair: [base_code, other_code]
        - cosine_distance: float
        - euclidean_distance: float
        - temporal_correlation: float
        - spectral_correlation: float
        - base_codebook_type: "active" o "inactive"
        - target_codebook_type: "active" o "inactive"
    """
    
    num_codes = all_embeddings.shape[0]
    
    # Determina tipo codebook del base
    base_type = get_codebook_type(base_code_idx, split_idx)
    
    print(f"   🔵 Decodificando codice base {base_code_idx} ({base_type})...")
    base_trace = decode_code_dual(model, base_code_idx, base_type, device, temporal_length)
    
    # Calcola distanze
    base_embedding = all_embeddings[base_code_idx:base_code_idx+1]
    
    # Cosine distances
    similarity_to_base = cosine_similarity(all_embeddings, base_embedding).flatten()
    cosine_distances = 1 - similarity_to_base
    
    # Euclidean distances
    euclidean_dists = euclidean_distances(all_embeddings, base_embedding).flatten()
    
    # Prepara altri codici
    code_indices = np.arange(num_codes)
    other_codes = code_indices[code_indices != base_code_idx]
    
    print(f"   📊 Decodificando {len(other_codes)} codici in batch (batch_size={batch_size})...")
    other_traces = decode_codes_batch_dual(model, other_codes, split_idx, device, temporal_length, batch_size=batch_size)
    
    # Calcola correlazioni
    print(f"   📊 Calcolando correlazioni per {len(other_codes)} coppie...")
    
    pairs_data = []
    
    for idx, (code_idx, trace) in enumerate(zip(other_codes, other_traces)):
        if (idx + 1) % 10 == 0:
            print(f"      Processate {idx + 1}/{len(other_codes)} coppie...")
        
        target_type = get_codebook_type(code_idx, split_idx)
        
        temp_corr = compute_temporal_correlation(base_trace, trace)
        spec_corr = compute_spectral_correlation(base_trace, trace)
        
        pair_data = {
            'code_pair': [int(base_code_idx), int(code_idx)],
            'cosine_distance': float(cosine_distances[code_idx]),
            'euclidean_distance': float(euclidean_dists[code_idx]),
            'temporal_correlation': float(temp_corr) if not np.isnan(temp_corr) else None,
            'spectral_correlation': float(spec_corr) if not np.isnan(spec_corr) else None,
            'base_codebook_type': base_type,
            'target_codebook_type': target_type,
        }
        
        pairs_data.append(pair_data)
    
    # Aggiungi anche la coppia base-base (distanza 0, correlazione 1)
    base_pair_data = {
        'code_pair': [int(base_code_idx), int(base_code_idx)],
        'cosine_distance': 0.0,
        'euclidean_distance': 0.0,
        'temporal_correlation': 1.0,
        'spectral_correlation': 1.0,
        'base_codebook_type': base_type,
        'target_codebook_type': base_type,
    }
    pairs_data.insert(0, base_pair_data)
    
    print(f"   ✅ {len(pairs_data)} coppie analizzate!")
    
    return pairs_data

# ============================================================================
# FUNZIONI - VISUALIZZAZIONE
# ============================================================================

def plot_correlation_vs_distance_dual(pairs_data, base_code_idx, output_dir):
    """Crea scatter plots di correlazione vs distanza, separati per tipo di coppia"""
    
    # Separa coppie per tipo
    active_active = [p for p in pairs_data if p['base_codebook_type'] == 'active' and p['target_codebook_type'] == 'active']
    active_inactive = [p for p in pairs_data if p['base_codebook_type'] == 'active' and p['target_codebook_type'] == 'inactive']
    inactive_active = [p for p in pairs_data if p['base_codebook_type'] == 'inactive' and p['target_codebook_type'] == 'active']
    inactive_inactive = [p for p in pairs_data if p['base_codebook_type'] == 'inactive' and p['target_codebook_type'] == 'inactive']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Correlazione vs Distanza - Base Code {base_code_idx} (DUAL CONSTRAINED)', fontsize=16)
    
    # Temporal correlation vs Cosine distance
    ax = axes[0, 0]
    for pair_type, pairs, label, color in [
        (active_active, active_active, 'Active-Active', 'blue'),
        (active_inactive, active_inactive, 'Active-Inactive', 'orange'),
        (inactive_active, inactive_active, 'Inactive-Active', 'green'),
        (inactive_inactive, inactive_inactive, 'Inactive-Inactive', 'red'),
    ]:
        if pairs:
            cos_dists = [p['cosine_distance'] for p in pairs if p['temporal_correlation'] is not None]
            temp_corrs = [p['temporal_correlation'] for p in pairs if p['temporal_correlation'] is not None]
            if cos_dists:
                ax.scatter(cos_dists, temp_corrs, label=label, alpha=0.6, s=20, c=color)
    ax.set_xlabel('Cosine Distance')
    ax.set_ylabel('Temporal Correlation')
    ax.set_title('Temporal Correlation vs Cosine Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temporal correlation vs Euclidean distance
    ax = axes[0, 1]
    for pair_type, pairs, label, color in [
        (active_active, active_active, 'Active-Active', 'blue'),
        (active_inactive, active_inactive, 'Active-Inactive', 'orange'),
        (inactive_active, inactive_active, 'Inactive-Active', 'green'),
        (inactive_inactive, inactive_inactive, 'Inactive-Inactive', 'red'),
    ]:
        if pairs:
            euc_dists = [p['euclidean_distance'] for p in pairs if p['temporal_correlation'] is not None]
            temp_corrs = [p['temporal_correlation'] for p in pairs if p['temporal_correlation'] is not None]
            if euc_dists:
                ax.scatter(euc_dists, temp_corrs, label=label, alpha=0.6, s=20, c=color)
    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('Temporal Correlation')
    ax.set_title('Temporal Correlation vs Euclidean Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Spectral correlation vs Cosine distance
    ax = axes[1, 0]
    for pair_type, pairs, label, color in [
        (active_active, active_active, 'Active-Active', 'blue'),
        (active_inactive, active_inactive, 'Active-Inactive', 'orange'),
        (inactive_active, inactive_active, 'Inactive-Active', 'green'),
        (inactive_inactive, inactive_inactive, 'Inactive-Inactive', 'red'),
    ]:
        if pairs:
            cos_dists = [p['cosine_distance'] for p in pairs if p['spectral_correlation'] is not None]
            spec_corrs = [p['spectral_correlation'] for p in pairs if p['spectral_correlation'] is not None]
            if cos_dists:
                ax.scatter(cos_dists, spec_corrs, label=label, alpha=0.6, s=20, c=color)
    ax.set_xlabel('Cosine Distance')
    ax.set_ylabel('Spectral Correlation')
    ax.set_title('Spectral Correlation vs Cosine Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Spectral correlation vs Euclidean distance
    ax = axes[1, 1]
    for pair_type, pairs, label, color in [
        (active_active, active_active, 'Active-Active', 'blue'),
        (active_inactive, active_inactive, 'Active-Inactive', 'orange'),
        (inactive_active, inactive_active, 'Inactive-Active', 'green'),
        (inactive_inactive, inactive_inactive, 'Inactive-Inactive', 'red'),
    ]:
        if pairs:
            euc_dists = [p['euclidean_distance'] for p in pairs if p['spectral_correlation'] is not None]
            spec_corrs = [p['spectral_correlation'] for p in pairs if p['spectral_correlation'] is not None]
            if euc_dists:
                ax.scatter(euc_dists, spec_corrs, label=label, alpha=0.6, s=20, c=color)
    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('Spectral Correlation')
    ax.set_title('Spectral Correlation vs Euclidean Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"correlation_distance_base_{base_code_idx}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath

# ============================================================================
# FUNZIONI - SALVATAGGIO JSON
# ============================================================================

def save_pairs_data_json(pairs_data, output_dir, base_code_idx, model_config):
    """Salva i dati delle coppie in JSON"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "coppie codice"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "sommario coppie"), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File principale con tutti i dati
    filename = f"coppie_codice_{base_code_idx}.json"
    filepath = os.path.join(output_dir, "coppie codice", filename)
    
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'base_code': base_code_idx,
            'num_pairs': len(pairs_data),
            'model_config': model_config,
            'checkpoint': CHECKPOINT_PATH,
            'model_type': 'DualCalciumVQVAE_Constrained',
        },
        'pairs': pairs_data
    }
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"   ✅ Dati salvati in: {filepath}")
    
    # Salva anche un summary con statistiche aggregate
    summary_filename = f"summary_codice_{base_code_idx}.json"
    summary_filepath = os.path.join(output_dir, "sommario coppie", summary_filename)
    
    # Calcola statistiche per tipo di coppia
    temporal_corrs = [p['temporal_correlation'] for p in pairs_data if p['temporal_correlation'] is not None]
    spectral_corrs = [p['spectral_correlation'] for p in pairs_data if p['spectral_correlation'] is not None]
    cosine_dists = [p['cosine_distance'] for p in pairs_data]
    euclidean_dists = [p['euclidean_distance'] for p in pairs_data]
    
    # Statistiche per tipo di coppia
    pair_type_stats = {}
    for pair_type in ['active-active', 'active-inactive', 'inactive-active', 'inactive-inactive']:
        base_type, target_type = pair_type.split('-')
        filtered_pairs = [p for p in pairs_data if p['base_codebook_type'] == base_type and p['target_codebook_type'] == target_type]
        
        if filtered_pairs:
            temp_corrs_type = [p['temporal_correlation'] for p in filtered_pairs if p['temporal_correlation'] is not None]
            spec_corrs_type = [p['spectral_correlation'] for p in filtered_pairs if p['spectral_correlation'] is not None]
            
            pair_type_stats[pair_type] = {
                'count': len(filtered_pairs),
                'temporal_correlation': {
                    'mean': float(np.mean(temp_corrs_type)) if temp_corrs_type else None,
                    'std': float(np.std(temp_corrs_type)) if temp_corrs_type else None,
                },
                'spectral_correlation': {
                    'mean': float(np.mean(spec_corrs_type)) if spec_corrs_type else None,
                    'std': float(np.std(spec_corrs_type)) if spec_corrs_type else None,
                },
            }
    
    summary_data = {
        'metadata': {
            'timestamp': timestamp,
            'base_code': base_code_idx,
            'num_pairs': len(pairs_data),
        },
        'statistics': {
            'temporal_correlation': {
                'mean': float(np.mean(temporal_corrs)) if temporal_corrs else None,
                'std': float(np.std(temporal_corrs)) if temporal_corrs else None,
                'min': float(np.min(temporal_corrs)) if temporal_corrs else None,
                'max': float(np.max(temporal_corrs)) if temporal_corrs else None,
                'valid_pairs': len(temporal_corrs),
            },
            'spectral_correlation': {
                'mean': float(np.mean(spectral_corrs)) if spectral_corrs else None,
                'std': float(np.std(spectral_corrs)) if spectral_corrs else None,
                'min': float(np.min(spectral_corrs)) if spectral_corrs else None,
                'max': float(np.max(spectral_corrs)) if spectral_corrs else None,
                'valid_pairs': len(spectral_corrs),
            },
            'cosine_distance': {
                'mean': float(np.mean(cosine_dists)),
                'std': float(np.std(cosine_dists)),
                'min': float(np.min(cosine_dists)),
                'max': float(np.max(cosine_dists)),
            },
            'euclidean_distance': {
                'mean': float(np.mean(euclidean_dists)),
                'std': float(np.std(euclidean_dists)),
                'min': float(np.min(euclidean_dists)),
                'max': float(np.max(euclidean_dists)),
            },
            'by_pair_type': pair_type_stats,
        }
    }
    
    with open(summary_filepath, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"   ✅ Summary salvato in: {summary_filepath}")
    
    return filepath, summary_filepath

# ============================================================================
# MAIN
# ============================================================================

def main():
    global ALL_BASE_CODES
    
    print("="*70)
    print("📊 ANALISI CORRELAZIONE vs DISTANZA - DUAL CONSTRAINED")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Temporal length: {TEMPORAL_LENGTH}")
    print(f"   Base codes da analizzare: {len(ALL_BASE_CODES)} ({ALL_BASE_CODES[0]} to {ALL_BASE_CODES[-1]})")
    print(f"   Batch size (decoding): {BATCH_SIZE}")
    print(f"   Output directory: {OUTPUT_DIR}/")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Pulisci cache GPU se disponibile
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("🧹 Cache GPU pulita")
    
    # Inizializza W&B
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "checkpoint_path": CHECKPOINT_PATH,
            "temporal_length": TEMPORAL_LENGTH,
            "batch_size": BATCH_SIZE,
            "num_base_codes": len(ALL_BASE_CODES),
            **MODEL_CONFIG
        },
        tags=["dual", "constrained", "correlation", "distance", "codebook"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # 1. Carica modello dual
        model, device = load_dual_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        print(f"💻 Device finale: {device}")
        
        # 2. Estrai embeddings dual
        active_emb, inactive_emb, all_embeddings, split_idx = extract_dual_codebook_embeddings(model)
        num_codes = all_embeddings.shape[0]
        
        # Aggiorna ALL_BASE_CODES se necessario
        if len(ALL_BASE_CODES) != num_codes:
            ALL_BASE_CODES = list(range(num_codes))
            print(f"   ⚠️  ALL_BASE_CODES aggiornato a: {len(ALL_BASE_CODES)} codici")
        
        print(f"\n🔍 Totale codici nel codebook: {num_codes}")
        print(f"   Active: {len(active_emb)}")
        print(f"   Inactive: {len(inactive_emb)}")
        print(f"   Coppie per codice base: {num_codes}")
        print(f"   Totale coppie da analizzare: {len(ALL_BASE_CODES) * num_codes}")
        
        # 3. Loop su tutti i base codes
        all_results = []
        
        for idx, base_code in enumerate(ALL_BASE_CODES):
            print(f"\n{'='*70}")
            print(f"📊 ANALISI CODICE BASE {base_code} ({idx+1}/{len(ALL_BASE_CODES)})")
            print(f"{'='*70}")
            
            # Analizza tutte le coppie per questo base code
            pairs_data = analyze_dual_code_pairs(
                model, all_embeddings, split_idx, device, TEMPORAL_LENGTH, base_code, batch_size=BATCH_SIZE
            )
            
            # Salva JSON per questo base code
            print(f"\n💾 Salvataggio dati per base code {base_code}...")
            main_file, summary_file = save_pairs_data_json(
                pairs_data, OUTPUT_DIR, base_code, MODEL_CONFIG
            )
            
            # Crea visualizzazioni
            print(f"📊 Creando visualizzazioni per base code {base_code}...")
            plot_file = plot_correlation_vs_distance_dual(pairs_data, base_code, OUTPUT_DIR)
            
            # Log su W&B
            wandb.log({
                f"correlation_distance/base_{base_code}/plot": wandb.Image(plot_file),
            })
            
            # Statistiche rapide
            temporal_corrs = [p['temporal_correlation'] for p in pairs_data if p['temporal_correlation'] is not None]
            spectral_corrs = [p['spectral_correlation'] for p in pairs_data if p['spectral_correlation'] is not None]
            
            # Statistiche per tipo di coppia
            for pair_type in ['active-active', 'active-inactive', 'inactive-active', 'inactive-inactive']:
                base_type, target_type = pair_type.split('-')
                filtered_pairs = [p for p in pairs_data if p['base_codebook_type'] == base_type and p['target_codebook_type'] == target_type]
                
                if filtered_pairs:
                    temp_corrs_type = [p['temporal_correlation'] for p in filtered_pairs if p['temporal_correlation'] is not None]
                    spec_corrs_type = [p['spectral_correlation'] for p in filtered_pairs if p['spectral_correlation'] is not None]
                    
                    if temp_corrs_type:
                        wandb.log({
                            f"correlation_distance/base_{base_code}/{pair_type}/temporal_corr_mean": np.mean(temp_corrs_type),
                            f"correlation_distance/base_{base_code}/{pair_type}/temporal_corr_std": np.std(temp_corrs_type),
                        })
                    
                    if spec_corrs_type:
                        wandb.log({
                            f"correlation_distance/base_{base_code}/{pair_type}/spectral_corr_mean": np.mean(spec_corrs_type),
                            f"correlation_distance/base_{base_code}/{pair_type}/spectral_corr_std": np.std(spec_corrs_type),
                        })
            
            print(f"   ✅ Base {base_code}: Temp corr mean={np.mean(temporal_corrs):.3f}, Spec corr mean={np.mean(spectral_corrs):.3f}")
            
            all_results.append({
                'base_code': base_code,
                'main_file': main_file,
                'summary_file': summary_file,
                'plot_file': plot_file,
                'num_pairs': len(pairs_data)
            })
            
            # Pulisci cache dopo ogni base code
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        # 4. Stampa riepilogo finale
        print(f"\n{'='*70}")
        print("✅ ANALISI COMPLETATA PER TUTTI I CODICI!")
        print(f"{'='*70}")
        print(f"\n📁 File salvati in: {OUTPUT_DIR}/")
        print(f"   - Totale base codes analizzati: {len(ALL_BASE_CODES)}")
        print(f"   - File JSON principali: {len(all_results)}")
        print(f"   - File summary: {len(all_results)}")
        print(f"   - File plot: {len(all_results)}")
        print(f"\n📋 Riepilogo file generati:")
        for result in all_results:
            print(f"   Base {result['base_code']:2d}: {os.path.basename(result['main_file'])}")
        
        wandb.finish()
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()


if __name__ == "__main__":
    main()

