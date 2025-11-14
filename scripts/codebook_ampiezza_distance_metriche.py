#!/usr/bin/env python3
"""
Analisi Ampiezza Codici - JSON Export Version

✨ CARATTERISTICHE:
- Calcola l'ampiezza (magnitude) di ogni codice nel codebook
- Calcola la differenza di ampiezza per tutte le coppie di codici
- Salva tutto in JSON locali
- NO grafici, NO W&B - solo dati puri

Output JSON contiene per ogni coppia:
- code_pair: [base_code, other_code]
- base_amplitude: float (magnitude del codice base)
- target_amplitude: float (magnitude del codice target)
- amplitude_difference_abs: float (differenza assoluta)
- amplitude_difference_rel: float (differenza relativa in %)

Uso:
    python codebook_amplitude_data_export.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from datetime import datetime

from models.vqvae import CalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/best_dual_model_32.pth"

# Directory output
OUTPUT_DIR = "ampiezza_coppie_dual"

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

# ✨ Automaticamente analizza TUTTI i codici come base
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
# FUNZIONI - CALCOLO AMPIEZZE
# ============================================================================

def compute_magnitudes(embeddings):
    """
    Calcola la magnitude (L2 norm) di ogni embedding
    
    Args:
        embeddings: array (num_codes, embedding_dim)
    
    Returns:
        magnitudes: array (num_codes,)
    """
    magnitudes = np.linalg.norm(embeddings, axis=1)
    
    print(f"\n📊 Statistiche Ampiezze:")
    print(f"   Mean: {magnitudes.mean():.4f}")
    print(f"   Std:  {magnitudes.std():.4f}")
    print(f"   Min:  {magnitudes.min():.4f} (code {magnitudes.argmin()})")
    print(f"   Max:  {magnitudes.max():.4f} (code {magnitudes.argmax()})")
    
    return magnitudes


def analyze_amplitude_pairs(embeddings, magnitudes, base_code_idx):
    """
    Analizza tutte le coppie di codici per le ampiezze (base vs others)
    
    Returns:
        List of dicts, ognuno contenente:
        - code_pair: [base_code, other_code]
        - base_amplitude: float
        - target_amplitude: float
        - amplitude_difference_abs: float
        - amplitude_difference_rel: float (%)
    """
    
    num_codes = embeddings.shape[0]
    
    base_amplitude = magnitudes[base_code_idx]
    
    pairs_data = []
    
    for target_code_idx in range(num_codes):
        target_amplitude = magnitudes[target_code_idx]
        
        # Differenza assoluta
        amp_diff_abs = abs(base_amplitude - target_amplitude)
        
        # Differenza relativa (percentuale rispetto al base)
        if base_amplitude > 1e-8:
            amp_diff_rel = (amp_diff_abs / base_amplitude) * 100.0
        else:
            amp_diff_rel = 0.0
        
        pair_data = {
            'code_pair': [int(base_code_idx), int(target_code_idx)],
            'base_amplitude': float(base_amplitude),
            'target_amplitude': float(target_amplitude),
            'amplitude_difference_abs': float(amp_diff_abs),
            'amplitude_difference_rel': float(amp_diff_rel),
        }
        
        pairs_data.append(pair_data)
    
    print(f"   ✅ {len(pairs_data)} coppie analizzate per base code {base_code_idx}")
    
    return pairs_data

# ============================================================================
# FUNZIONI - SALVATAGGIO JSON
# ============================================================================

def save_amplitude_pairs_json(pairs_data, output_dir, base_code_idx, model_config):
    """Salva i dati delle ampiezze delle coppie in JSON"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File principale con tutti i dati
    filename = f"ampiezza_coppie_codice_{base_code_idx}.json"
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
    summary_filename = f"summary_ampiezza_codice_{base_code_idx}.json"
    summary_filepath = os.path.join(output_dir, summary_filename)
    
    # Calcola statistiche
    amp_diffs_abs = [p['amplitude_difference_abs'] for p in pairs_data]
    amp_diffs_rel = [p['amplitude_difference_rel'] for p in pairs_data]
    base_amp = pairs_data[0]['base_amplitude']
    target_amps = [p['target_amplitude'] for p in pairs_data]
    
    summary_data = {
        'metadata': {
            'timestamp': timestamp,
            'base_code': base_code_idx,
            'num_pairs': len(pairs_data),
        },
        'base_code_amplitude': float(base_amp),
        'statistics': {
            'amplitude_difference_abs': {
                'mean': float(np.mean(amp_diffs_abs)),
                'std': float(np.std(amp_diffs_abs)),
                'min': float(np.min(amp_diffs_abs)),
                'max': float(np.max(amp_diffs_abs)),
            },
            'amplitude_difference_rel': {
                'mean': float(np.mean(amp_diffs_rel)),
                'std': float(np.std(amp_diffs_rel)),
                'min': float(np.min(amp_diffs_rel)),
                'max': float(np.max(amp_diffs_rel)),
            },
            'target_amplitudes': {
                'mean': float(np.mean(target_amps)),
                'std': float(np.std(target_amps)),
                'min': float(np.min(target_amps)),
                'max': float(np.max(target_amps)),
            },
        }
    }
    
    with open(summary_filepath, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"   ✅ Summary salvato in: {summary_filepath}")
    
    return filepath, summary_filepath


def save_all_magnitudes_json(magnitudes, output_dir, model_config):
    """Salva tutte le ampiezze individuali in un unico file"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "all_code_magnitudes.json"
    filepath = os.path.join(output_dir, filename)
    
    magnitudes_data = {
        'metadata': {
            'timestamp': timestamp,
            'num_codes': len(magnitudes),
            'model_config': model_config,
            'checkpoint': CHECKPOINT_PATH,
        },
        'magnitudes': [
            {
                'code_index': int(i),
                'amplitude': float(mag)
            }
            for i, mag in enumerate(magnitudes)
        ],
        'statistics': {
            'mean': float(np.mean(magnitudes)),
            'std': float(np.std(magnitudes)),
            'min': float(np.min(magnitudes)),
            'max': float(np.max(magnitudes)),
            'min_code': int(np.argmin(magnitudes)),
            'max_code': int(np.argmax(magnitudes)),
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(magnitudes_data, f, indent=2)
    
    print(f"\n✅ Tutte le ampiezze salvate in: {filepath}")
    
    return filepath


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("📊 ANALISI AMPIEZZE CODICI - JSON Export Version")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Base codes da analizzare: {len(ALL_BASE_CODES)} ({ALL_BASE_CODES[0]} to {ALL_BASE_CODES[-1]})")
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
        
        # 3. Calcola tutte le ampiezze
        print(f"\n{'='*70}")
        print("📏 CALCOLO AMPIEZZE (L2 NORM)")
        print(f"{'='*70}")
        
        magnitudes = compute_magnitudes(embeddings)
        
        # 4. Salva tutte le ampiezze individuali
        all_mags_file = save_all_magnitudes_json(magnitudes, OUTPUT_DIR, MODEL_CONFIG)
        
        # 5. Loop su tutti i base codes per calcolare differenze
        print(f"\n{'='*70}")
        print("📊 CALCOLO DIFFERENZE DI AMPIEZZA PER TUTTE LE COPPIE")
        print(f"{'='*70}")
        print(f"   Coppie per codice base: {num_codes}")
        print(f"   Totale coppie da analizzare: {len(ALL_BASE_CODES) * num_codes}")
        
        all_results = []
        
        for idx, base_code in enumerate(ALL_BASE_CODES):
            print(f"\n{'='*70}")
            print(f"📊 BASE CODE {base_code} ({idx+1}/{len(ALL_BASE_CODES)})")
            print(f"{'='*70}")
            
            # Analizza tutte le coppie per questo base code
            pairs_data = analyze_amplitude_pairs(embeddings, magnitudes, base_code)
            
            # Salva JSON per questo base code
            print(f"   💾 Salvataggio dati per base code {base_code}...")
            main_file, summary_file = save_amplitude_pairs_json(
                pairs_data, OUTPUT_DIR, base_code, MODEL_CONFIG
            )
            
            # Statistiche rapide
            amp_diffs = [p['amplitude_difference_abs'] for p in pairs_data]
            
            print(f"   ✅ Base {base_code}: Amp={magnitudes[base_code]:.3f}, "
                  f"Mean diff={np.mean(amp_diffs):.3f}, Max diff={np.max(amp_diffs):.3f}")
            
            all_results.append({
                'base_code': base_code,
                'main_file': main_file,
                'summary_file': summary_file,
                'num_pairs': len(pairs_data)
            })
        
        # 6. Stampa riepilogo finale
        print(f"\n{'='*70}")
        print("✅ ANALISI COMPLETATA PER TUTTI I CODICI!")
        print(f"{'='*70}")
        print(f"\n📁 File salvati in: {OUTPUT_DIR}/")
        print(f"   - Totale base codes analizzati: {len(ALL_BASE_CODES)}")
        print(f"   - File JSON coppie: {len(all_results)}")
        print(f"   - File summary: {len(all_results)}")
        print(f"   - File ampiezze individuali: {all_mags_file}")
        print(f"\n📋 Riepilogo file generati:")
        for result in all_results:
            print(f"   Base {result['base_code']:2d}: {os.path.basename(result['main_file'])}")
        
        print(f"\n💡 PROSSIMI PASSI:")
        print(f"   1. Usa questi dati JSON per creare scatter plots")
        print(f"   2. Combina con i dati di correlazione/distanza")
        print(f"   3. Visualizza la relazione: Distanza vs Differenza Ampiezza")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()