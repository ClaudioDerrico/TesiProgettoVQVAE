#!/usr/bin/env python3
"""
Analisi Ampiezza Codici - JSON Export Version (DUAL CODEBOOK)

✨ CARATTERISTICHE:
- Calcola l'ampiezza (magnitude) di ogni codice nel DUAL codebook
- Separazione tra codici ACTIVE e INACTIVE
- Calcola la differenza di ampiezza per tutte le coppie di codici
- Salva tutto in JSON locali
- NO grafici, NO W&B - solo dati puri

Output JSON contiene per ogni coppia:
- code_pair: [base_code, other_code]
- base_amplitude: float (magnitude del codice base)
- target_amplitude: float (magnitude del codice target)
- amplitude_difference_abs: float (differenza assoluta)
- amplitude_difference_rel: float (differenza relativa in %)
- base_codebook_type: "active" o "inactive"
- target_codebook_type: "active" o "inactive"

Uso:
    python codebook_amplitude_data_export_dual.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from datetime import datetime

from models.dual_vqvae import DualCalciumVQVAE

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/dual_codebook/best_dual_model_32.pth"

# Directory output
OUTPUT_DIR = "ampiezza_coppie_dual"

MODEL_CONFIG = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 32,      # 512 total = 256 active + 256 inactive
    'embedding_dim': 32,
    'commitment_cost': 0.25,
    'dropout_rate': 0.0,
    'use_quantizer': True,
    'threshold': 0.0,
    'threshold_type': 'adaptive',
}

# ✨ Automaticamente analizza TUTTI i codici come base
ALL_BASE_CODES = list(range(32))  # [0, 1, 2, ..., 511]

# ============================================================================
# FUNZIONI - CARICAMENTO MODELLO DUAL
# ============================================================================

def load_dual_model(checkpoint_path, config, device):
    """Carica il modello dual codebook allenato"""
    print(f"🔄 Caricando modello DUAL CODEBOOK da: {checkpoint_path}")
    
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
        threshold=config['threshold'],
        threshold_type=config['threshold_type']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        best_corr = checkpoint.get('best_correlation', 'N/A')
        print(f"✅ Modello caricato (epoca {epoch}, best_corr={best_corr})")
        
        # Stampa statistiche del codebook se disponibili
        if 'codebook_stats' in checkpoint:
            stats = checkpoint['codebook_stats']
            print(f"\n📚 Statistiche Codebook dal checkpoint:")
            print(f"   Total usage: {stats.get('total_usage_percentage', 'N/A')}%")
            print(f"   Active usage: {stats.get('active_usage_percentage', 'N/A')}%")
            print(f"   Inactive usage: {stats.get('inactive_usage_percentage', 'N/A')}%")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Modello caricato")
    
    model = model.to(device)
    model.eval()
    
    return model

def extract_dual_codebook_embeddings(model):
    """
    Estrae gli embeddings del DUAL codebook
    
    Returns:
        active_embeddings: array (num_active_codes, embedding_dim)
        inactive_embeddings: array (num_inactive_codes, embedding_dim)
        split_idx: indice di separazione
        all_embeddings: array completo
    """
    # ✅ CORREZIONE: sono embedding_active e embedding_inactive
    active_embeddings = model.vector_quantization.embedding_active.weight.data.cpu().numpy()
    inactive_embeddings = model.vector_quantization.embedding_inactive.weight.data.cpu().numpy()
    
    # Concatena per avere tutti gli embeddings
    all_embeddings = np.concatenate([active_embeddings, inactive_embeddings], axis=0)
    split_idx = len(active_embeddings)
    
    print(f"✅ Dual Codebook embeddings estratti:")
    print(f"   Active embeddings: {active_embeddings.shape}")
    print(f"   Inactive embeddings: {inactive_embeddings.shape}")
    print(f"   Total shape: {all_embeddings.shape}")
    print(f"   Split index: {split_idx}")
    
    return active_embeddings, inactive_embeddings, split_idx, all_embeddings

# ============================================================================
# FUNZIONI - CALCOLO AMPIEZZE DUAL
# ============================================================================

def compute_dual_magnitudes(active_embeddings, inactive_embeddings):
    """
    Calcola la magnitude (L2 norm) per entrambi i codebook
    
    Returns:
        dict con statistiche separate per active e inactive
    """
    active_mags = np.linalg.norm(active_embeddings, axis=1)
    inactive_mags = np.linalg.norm(inactive_embeddings, axis=1)
    
    all_mags = np.concatenate([active_mags, inactive_mags])
    
    print(f"\n📊 Statistiche Ampiezze DUAL CODEBOOK:")
    print(f"\n🟢 ACTIVE Codebook:")
    print(f"   Mean: {active_mags.mean():.4f}")
    print(f"   Std:  {active_mags.std():.4f}")
    print(f"   Min:  {active_mags.min():.4f} (code {active_mags.argmin()})")
    print(f"   Max:  {active_mags.max():.4f} (code {active_mags.argmax()})")
    
    print(f"\n🔴 INACTIVE Codebook:")
    print(f"   Mean: {inactive_mags.mean():.4f}")
    print(f"   Std:  {inactive_mags.std():.4f}")
    print(f"   Min:  {inactive_mags.min():.4f} (code {inactive_mags.argmin()})")
    print(f"   Max:  {inactive_mags.max():.4f} (code {inactive_mags.argmax()})")
    
    print(f"\n📊 OVERALL:")
    print(f"   Mean: {all_mags.mean():.4f}")
    print(f"   Std:  {all_mags.std():.4f}")
    print(f"   Min:  {all_mags.min():.4f}")
    print(f"   Max:  {all_mags.max():.4f}")
    
    return {
        'active_magnitudes': active_mags,
        'inactive_magnitudes': inactive_mags,
        'all_magnitudes': all_mags,
    }


def get_codebook_type(code_idx, split_idx):
    """Determina se un codice è active o inactive"""
    return "active" if code_idx < split_idx else "inactive"


def analyze_dual_amplitude_pairs(all_embeddings, magnitudes, base_code_idx, split_idx):
    """
    Analizza tutte le coppie di codici per le ampiezze (base vs others)
    Con informazioni su quale codebook appartiene
    
    Returns:
        List of dicts, ognuno contenente:
        - code_pair: [base_code, other_code]
        - base_amplitude: float
        - target_amplitude: float
        - amplitude_difference_abs: float
        - amplitude_difference_rel: float (%)
        - base_codebook_type: "active" o "inactive"
        - target_codebook_type: "active" o "inactive"
    """
    
    num_codes = all_embeddings.shape[0]
    all_mags = magnitudes['all_magnitudes']
    
    base_amplitude = all_mags[base_code_idx]
    base_type = get_codebook_type(base_code_idx, split_idx)
    
    pairs_data = []
    
    for target_code_idx in range(num_codes):
        target_amplitude = all_mags[target_code_idx]
        target_type = get_codebook_type(target_code_idx, split_idx)
        
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
            'base_codebook_type': base_type,
            'target_codebook_type': target_type,
        }
        
        pairs_data.append(pair_data)
    
    print(f"   ✅ {len(pairs_data)} coppie analizzate per base code {base_code_idx} ({base_type})")
    
    return pairs_data

# ============================================================================
# FUNZIONI - SALVATAGGIO JSON
# ============================================================================

def save_dual_amplitude_pairs_json(pairs_data, output_dir, base_code_idx, model_config, split_idx):
    """Salva i dati delle ampiezze delle coppie in JSON - SOLO FILE PRINCIPALE"""
    
    # Crea sottodirectory "coppie codice"
    output_subdir = os.path.join(output_dir, "coppie codice")
    os.makedirs(output_subdir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File principale con tutti i dati
    filename = f"ampiezza_coppie_codice_{base_code_idx}.json"
    filepath = os.path.join(output_subdir, filename)
    
    base_type = get_codebook_type(base_code_idx, split_idx)
    
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'base_code': base_code_idx,
            'base_codebook_type': base_type,
            'num_pairs': len(pairs_data),
            'model_config': model_config,
            'checkpoint': CHECKPOINT_PATH,
            'split_index': split_idx,
        },
        'pairs': pairs_data
    }
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"   ✅ Dati salvati in: {filepath}")
    
    return filepath


def save_all_dual_magnitudes_json(magnitudes, output_dir, model_config, split_idx):
    """Salva tutte le ampiezze individuali in un unico file"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "all_code_magnitudes_dual.json"
    filepath = os.path.join(output_dir, filename)
    
    all_mags = magnitudes['all_magnitudes']
    active_mags = magnitudes['active_magnitudes']
    inactive_mags = magnitudes['inactive_magnitudes']
    
    magnitudes_data = {
        'metadata': {
            'timestamp': timestamp,
            'num_codes': len(all_mags),
            'num_active': len(active_mags),
            'num_inactive': len(inactive_mags),
            'split_index': split_idx,
            'model_config': model_config,
            'checkpoint': CHECKPOINT_PATH,
        },
        'magnitudes': [
            {
                'code_index': int(i),
                'amplitude': float(mag),
                'codebook_type': get_codebook_type(i, split_idx)
            }
            for i, mag in enumerate(all_mags)
        ],
        'statistics': {
            'overall': {
                'mean': float(np.mean(all_mags)),
                'std': float(np.std(all_mags)),
                'min': float(np.min(all_mags)),
                'max': float(np.max(all_mags)),
                'min_code': int(np.argmin(all_mags)),
                'max_code': int(np.argmax(all_mags)),
            },
            'active': {
                'mean': float(np.mean(active_mags)),
                'std': float(np.std(active_mags)),
                'min': float(np.min(active_mags)),
                'max': float(np.max(active_mags)),
            },
            'inactive': {
                'mean': float(np.mean(inactive_mags)),
                'std': float(np.std(inactive_mags)),
                'min': float(np.min(inactive_mags)),
                'max': float(np.max(inactive_mags)),
            }
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
    print("📊 ANALISI AMPIEZZE DUAL CODEBOOK - JSON Export Version")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings TOTAL: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Num embeddings ACTIVE: {MODEL_CONFIG['num_embeddings']//2}")
    print(f"   Num embeddings INACTIVE: {MODEL_CONFIG['num_embeddings']//2}")
    print(f"   Base codes da analizzare: {len(ALL_BASE_CODES)} ({ALL_BASE_CODES[0]} to {ALL_BASE_CODES[-1]})")
    print(f"   Output directory: {OUTPUT_DIR}/")
    print(f"   ✨ Mode: JSON Export Only (no plots, no W&B, no summaries)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    try:
        # 1. Carica modello dual
        model = load_dual_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        
        # 2. Estrai embeddings dual
        active_emb, inactive_emb, split_idx, all_emb = extract_dual_codebook_embeddings(model)
        num_codes = all_emb.shape[0]
        
        print(f"\n🔍 Totale codici nel codebook: {num_codes}")
        print(f"   Active: {len(active_emb)}")
        print(f"   Inactive: {len(inactive_emb)}")
        
        # 3. Calcola tutte le ampiezze
        print(f"\n{'='*70}")
        print("📏 CALCOLO AMPIEZZE (L2 NORM) - DUAL CODEBOOK")
        print(f"{'='*70}")
        
        magnitudes = compute_dual_magnitudes(active_emb, inactive_emb)
        
        # 4. Salva tutte le ampiezze individuali
        all_mags_file = save_all_dual_magnitudes_json(magnitudes, OUTPUT_DIR, MODEL_CONFIG, split_idx)
        
        # 5. Loop su tutti i base codes per calcolare differenze
        print(f"\n{'='*70}")
        print("📊 CALCOLO DIFFERENZE DI AMPIEZZA PER TUTTE LE COPPIE")
        print(f"{'='*70}")
        print(f"   Coppie per codice base: {num_codes}")
        print(f"   Totale coppie da analizzare: {len(ALL_BASE_CODES) * num_codes}")
        
        all_results = []
        
        for idx, base_code in enumerate(ALL_BASE_CODES):
            base_type = get_codebook_type(base_code, split_idx)
            
            if (idx + 1) % 50 == 0:  # Progress ogni 50 codici
                print(f"\n{'='*70}")
                print(f"📊 Progress: {idx+1}/{len(ALL_BASE_CODES)} codici completati")
                print(f"{'='*70}")
            
            # Analizza tutte le coppie per questo base code
            pairs_data = analyze_dual_amplitude_pairs(all_emb, magnitudes, base_code, split_idx)
            
            # Salva JSON per questo base code
            main_file = save_dual_amplitude_pairs_json(
                pairs_data, OUTPUT_DIR, base_code, MODEL_CONFIG, split_idx
            )
            
            # Statistiche rapide
            amp_diffs = [p['amplitude_difference_abs'] for p in pairs_data]
            
            all_results.append({
                'base_code': base_code,
                'base_type': base_type,
                'main_file': main_file,
                'num_pairs': len(pairs_data),
                'mean_diff': np.mean(amp_diffs),
                'max_diff': np.max(amp_diffs)
            })
        
        # 6. Stampa riepilogo finale
        print(f"\n{'='*70}")
        print("✅ ANALISI COMPLETATA PER TUTTI I CODICI!")
        print(f"{'='*70}")
        print(f"\n📁 File salvati in: {OUTPUT_DIR}/coppie codice/")
        print(f"   - Totale base codes analizzati: {len(ALL_BASE_CODES)}")
        print(f"     • Active: {sum(1 for r in all_results if r['base_type'] == 'active')}")
        print(f"     • Inactive: {sum(1 for r in all_results if r['base_type'] == 'inactive')}")
        print(f"   - File JSON coppie: {len(all_results)}")
        print(f"   - File ampiezze individuali: {all_mags_file}")
        
        print(f"\n💡 PROSSIMI PASSI:")
        print(f"   1. Usa questi dati JSON per creare scatter plots")
        print(f"   2. Combina con i dati di correlazione/distanza")
        print(f"   3. Analizza differenze tra coppie active-active, active-inactive, etc.")
        print(f"   4. Visualizza la relazione: Distanza vs Differenza Ampiezza per tipo di coppia")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()