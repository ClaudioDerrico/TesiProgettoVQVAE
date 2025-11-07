#!/usr/bin/env python3
"""
Analisi Correlazione vs Distanza - JSON Export Version

✨ CARATTERISTICHE:
- Calcola correlazioni temporali e spettrali tra codice base e tutti gli altri
- Calcola distanze cosine ed euclidean
- Salva tutto in JSON locali
- NO grafici, NO W&B - solo dati puri

Output JSON contiene per ogni coppia:
- code_pair: [base_code, other_code]
- cosine_distance: float
- euclidean_distance: float
- temporal_correlation: float
- spectral_correlation: float

Uso:
    python codebook_correlation_distance_data_export.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr

from models.vqvae import CalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/best_single_neuron_model_32.pth"

# Directory output
OUTPUT_DIR = "correlazione_distanze_temporali_spettrali"

MODEL_CONFIG = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 32,
    'embedding_dim': 32,
    'commitment_cost': 2.0,
    'dropout_rate': 0.1,
    'use_quantizer': True,
}

# Parametri
TEMPORAL_LENGTH = 15
BATCH_SIZE = 8

# ✨ Automaticamente analizza TUTTI i codici come base
# Se vuoi solo alcuni codici specifici, modifica questa lista
ALL_BASE_CODES = list(range(32))  # [0, 1, 2, ..., 31]

# ============================================================================
# FUNZIONI - CARICAMENTO MODELLO
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

# ============================================================================
# FUNZIONI - DECODIFICA
# ============================================================================

def decode_code(model, code_idx, device, temporal_length):
    """Decodifica un singolo codice"""
    code_embedding = model.vector_quantization.embedding.weight[code_idx]
    sequence = code_embedding.unsqueeze(1).repeat(1, temporal_length)
    sequence = sequence.unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstruction = model.decode(sequence)
    
    trace = reconstruction.cpu().numpy()[0, 0, :]
    return trace


def decode_codes_batch(model, code_indices, device, temporal_length, batch_size=8):
    """Decodifica multipli codici in batch"""
    all_traces = []
    num_codes = len(code_indices)
    
    model.eval()
    with torch.no_grad():
        for i in range(0, num_codes, batch_size):
            batch_indices = code_indices[i:i+batch_size]
            batch_embeddings = model.vector_quantization.embedding.weight[batch_indices]
            batch_sequences = batch_embeddings.unsqueeze(2).repeat(1, 1, temporal_length)
            batch_sequences = batch_sequences.to(device)
            
            reconstructions = model.decode(batch_sequences)
            all_traces.append(reconstructions.cpu().numpy()[:, 0, :])
    
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

def analyze_code_pairs(model, embeddings, device, temporal_length, base_code_idx, batch_size=8):
    """
    Analizza tutte le coppie di codici (base vs others)
    
    Returns:
        List of dicts, ognuno contenente:
        - code_pair: [base_code, other_code]
        - cosine_distance: float
        - euclidean_distance: float
        - temporal_correlation: float
        - spectral_correlation: float
    """
    
    num_codes = embeddings.shape[0]
    
    print(f"   🔵 Decodificando codice base {base_code_idx}...")
    base_trace = decode_code(model, base_code_idx, device, temporal_length)
    
    # Calcola distanze
    base_embedding = embeddings[base_code_idx:base_code_idx+1]
    
    # Cosine distances
    similarity_to_base = cosine_similarity(embeddings, base_embedding).flatten()
    cosine_distances = 1 - similarity_to_base
    
    # Euclidean distances
    euclidean_dists = euclidean_distances(embeddings, base_embedding).flatten()
    
    # Prepara altri codici
    code_indices = np.arange(num_codes)
    other_codes = code_indices[code_indices != base_code_idx]
    
    print(f"   📊 Decodificando {len(other_codes)} codici in batch (batch_size={batch_size})...")
    other_traces = decode_codes_batch(model, other_codes, device, temporal_length, batch_size=batch_size)
    
    # Calcola correlazioni
    print(f"   📊 Calcolando correlazioni per {len(other_codes)} coppie...")
    
    pairs_data = []
    
    for idx, (code_idx, trace) in enumerate(zip(other_codes, other_traces)):
        if (idx + 1) % 10 == 0:
            print(f"      Processate {idx + 1}/{len(other_codes)} coppie...")
        
        temp_corr = compute_temporal_correlation(base_trace, trace)
        spec_corr = compute_spectral_correlation(base_trace, trace)
        
        pair_data = {
            'code_pair': [int(base_code_idx), int(code_idx)],
            'cosine_distance': float(cosine_distances[code_idx]),
            'euclidean_distance': float(euclidean_dists[code_idx]),
            'temporal_correlation': float(temp_corr) if not np.isnan(temp_corr) else None,
            'spectral_correlation': float(spec_corr) if not np.isnan(spec_corr) else None,
        }
        
        pairs_data.append(pair_data)
    
    # Aggiungi anche la coppia base-base (distanza 0, correlazione 1)
    base_pair_data = {
        'code_pair': [int(base_code_idx), int(base_code_idx)],
        'cosine_distance': 0.0,
        'euclidean_distance': 0.0,
        'temporal_correlation': 1.0,
        'spectral_correlation': 1.0,
    }
    pairs_data.insert(0, base_pair_data)
    
    print(f"   ✅ {len(pairs_data)} coppie analizzate!")
    
    return pairs_data

# ============================================================================
# FUNZIONI - SALVATAGGIO JSON
# ============================================================================

def save_pairs_data_json(pairs_data, output_dir, base_code_idx, model_config):
    """Salva i dati delle coppie in JSON"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File principale con tutti i dati
    filename = f"coppie_codice_{base_code_idx}.json"
    filepath = os.path.join(output_dir, filename)
    
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'base_code': base_code_idx,
            'num_pairs': len(pairs_data),
            'model_config': model_config,
            'checkpoint': CHECKPOINT_PATH,
        },
        'pairs': pairs_data
    }
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"   ✅ Dati salvati in: {filepath}")
    
    # Salva anche un summary con statistiche aggregate
    summary_filename = f"summary_codice_{base_code_idx}.json"
    summary_filepath = os.path.join(output_dir, summary_filename)
    
    # Calcola statistiche
    temporal_corrs = [p['temporal_correlation'] for p in pairs_data if p['temporal_correlation'] is not None]
    spectral_corrs = [p['spectral_correlation'] for p in pairs_data if p['spectral_correlation'] is not None]
    cosine_dists = [p['cosine_distance'] for p in pairs_data]
    euclidean_dists = [p['euclidean_distance'] for p in pairs_data]
    
    summary_data = {
        'metadata': {
            'timestamp': timestamp,
            'base_code': base_code_idx,
            'num_pairs': len(pairs_data),
        },
        'statistics': {
            'temporal_correlation': {
                'mean': float(np.mean(temporal_corrs)),
                'std': float(np.std(temporal_corrs)),
                'min': float(np.min(temporal_corrs)),
                'max': float(np.max(temporal_corrs)),
                'valid_pairs': len(temporal_corrs),
            },
            'spectral_correlation': {
                'mean': float(np.mean(spectral_corrs)),
                'std': float(np.std(spectral_corrs)),
                'min': float(np.min(spectral_corrs)),
                'max': float(np.max(spectral_corrs)),
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
    print("="*70)
    print("📊 ANALISI CORRELAZIONE vs DISTANZA - JSON Export Version")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Temporal length: {TEMPORAL_LENGTH}")
    print(f"   Base codes da analizzare: {len(ALL_BASE_CODES)} ({ALL_BASE_CODES[0]} to {ALL_BASE_CODES[-1]})")
    print(f"   Batch size (decoding): {BATCH_SIZE}")
    print(f"   Output directory: {OUTPUT_DIR}/")
    print(f"   ✨ Mode: JSON Export Only (no plots, no W&B)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    try:
        # 1. Carica modello
        model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # 2. Estrai embeddings
        embeddings = extract_codebook_embeddings(model)
        num_codes = embeddings.shape[0]
        
        print(f"\n🔍 Totale codici nel codebook: {num_codes}")
        print(f"   Coppie per codice base: {num_codes}")
        print(f"   Totale coppie da analizzare: {len(ALL_BASE_CODES) * num_codes}")
        
        # 3. Loop su tutti i base codes
        all_results = []
        
        for idx, base_code in enumerate(ALL_BASE_CODES):
            print(f"\n{'='*70}")
            print(f"📊 ANALISI CODICE BASE {base_code} ({idx+1}/{len(ALL_BASE_CODES)})")
            print(f"{'='*70}")
            
            # Analizza tutte le coppie per questo base code
            pairs_data = analyze_code_pairs(
                model, embeddings, device, TEMPORAL_LENGTH, base_code, batch_size=BATCH_SIZE
            )
            
            # Salva JSON per questo base code
            print(f"\n💾 Salvataggio dati per base code {base_code}...")
            main_file, summary_file = save_pairs_data_json(
                pairs_data, OUTPUT_DIR, base_code, MODEL_CONFIG
            )
            
            # Statistiche rapide
            temporal_corrs = [p['temporal_correlation'] for p in pairs_data if p['temporal_correlation'] is not None]
            spectral_corrs = [p['spectral_correlation'] for p in pairs_data if p['spectral_correlation'] is not None]
            
            print(f"   ✅ Base {base_code}: Temp corr mean={np.mean(temporal_corrs):.3f}, Spec corr mean={np.mean(spectral_corrs):.3f}")
            
            all_results.append({
                'base_code': base_code,
                'main_file': main_file,
                'summary_file': summary_file,
                'num_pairs': len(pairs_data)
            })
        
        # 4. Stampa riepilogo finale
        print(f"\n{'='*70}")
        print("✅ ANALISI COMPLETATA PER TUTTI I CODICI!")
        print(f"{'='*70}")
        print(f"\n📁 File salvati in: {OUTPUT_DIR}/")
        print(f"   - Totale base codes analizzati: {len(ALL_BASE_CODES)}")
        print(f"   - File JSON principali: {len(all_results)}")
        print(f"   - File summary: {len(all_results)}")
        print(f"\n📋 Riepilogo file generati:")
        for result in all_results:
            print(f"   Base {result['base_code']:2d}: {os.path.basename(result['main_file'])}")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()