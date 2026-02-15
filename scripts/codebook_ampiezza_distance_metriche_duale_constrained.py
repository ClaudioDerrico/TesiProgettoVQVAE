#!/usr/bin/env python3
"""
Analisi Ampiezza Codici - JSON Export Version (DUAL CODEBOOK CONSTRAINED)

✨ CARATTERISTICHE:
- Calcola l'ampiezza (magnitude) di ogni codice nel DUAL codebook CONSTRAINED
- Separazione tra codici ACTIVE e INACTIVE
- Calcola la differenza di ampiezza per tutte le coppie di codici
- Salva tutto in JSON locali
- Genera grafici e log su W&B

Output JSON contiene per ogni coppia:
- code_pair: [base_code, other_code]
- base_amplitude: float (magnitude del codice base)
- target_amplitude: float (magnitude del codice target)
- amplitude_difference_abs: float (differenza assoluta)
- amplitude_difference_rel: float (differenza relativa in %)
- base_codebook_type: "active" o "inactive"
- target_codebook_type: "active" o "inactive"

Uso:
    python scripts/codebook_ampiezza_distance_metriche_duale_constrained.py
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

from models.dual_vqvae import DualCalciumVQVAE

# ============================================================================
# ⚙️ CONFIGURAZIONE - MODIFICA QUI PER CAMBIARE MODELLO
# ============================================================================

# ✅ MODIFICA QUI: Path al checkpoint constrained
CHECKPOINT_PATH = "results/dual_codebook_constrained/best_dual_model_32_constrained.pth"

# Directory output
OUTPUT_DIR = "ampiezza_coppie_dual_constrained"

# W&B Configuration
WANDB_PROJECT = "calcium-vqvae-amplitude-distance-dual-constrained"
WANDB_RUN_NAME = f"amplitude-distance-dual-constrained-{datetime.now().strftime('%m%d-%H%M')}"

# ⚙️ CONFIGURAZIONE MODELLO: Deve corrispondere al checkpoint
MODEL_CONFIG = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 32,         # ⬅️ CAMBIA QUI: 16, 32, 64, 128, 256, 512, 1024, etc.
    'embedding_dim': 32,          # ⬅️ CAMBIA QUI: tipicamente = num_embeddings per piccoli, 64 per grandi
    'commitment_cost': 0.25,
    'dropout_rate': 0.0,
    'use_quantizer': True,
    'threshold': 0.0,
    'threshold_type': 'adaptive',
}

# ✨ Automaticamente analizza TUTTI i codici come base
# ⬅️ CAMBIA QUI: Deve corrispondere a num_embeddings
ALL_BASE_CODES = list(range(32))  # [0, 1, 2, ..., 31] per 32 embeddings

# ============================================================================
# FUNZIONI - CARICAMENTO MODELLO DUAL
# ============================================================================

def load_dual_model(checkpoint_path, config, device):
    """Carica il modello dual codebook CONSTRAINED allenato"""
    print(f"🔄 Caricando modello DUAL CODEBOOK CONSTRAINED da: {checkpoint_path}")
    
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
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        best_corr = checkpoint.get('best_correlation', 'N/A')
        print(f"✅ Modello CONSTRAINED caricato (epoca {epoch}, best_corr={best_corr})")
        
        # Stampa statistiche del codebook se disponibili
        if 'codebook_stats' in checkpoint:
            stats = checkpoint['codebook_stats']
            print(f"\n📚 Statistiche Codebook dal checkpoint:")
            print(f"   Total usage: {stats.get('total_usage_percentage', 'N/A')}%")
            print(f"   Active usage: {stats.get('active_usage_percentage', 'N/A')}%")
            print(f"   Inactive usage: {stats.get('inactive_usage_percentage', 'N/A')}%")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Modello CONSTRAINED caricato")
    
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
    
    print(f"✅ Dual Codebook CONSTRAINED embeddings estratti:")
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
    
    print(f"\n📊 Statistiche Ampiezze DUAL CODEBOOK CONSTRAINED:")
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
        - cosine_distance: float
        - euclidean_distance: float
    """
    
    num_codes = all_embeddings.shape[0]
    all_mags = magnitudes['all_magnitudes']
    
    base_amplitude = all_mags[base_code_idx]
    base_type = get_codebook_type(base_code_idx, split_idx)
    
    # Calcola distanze cosine ed euclidean
    base_embedding = all_embeddings[base_code_idx:base_code_idx+1]
    similarity_to_base = cosine_similarity(all_embeddings, base_embedding).flatten()
    cosine_distances = 1 - similarity_to_base
    euclidean_distances_arr = euclidean_distances(all_embeddings, base_embedding).flatten()
    
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
            'cosine_distance': float(cosine_distances[target_code_idx]),
            'euclidean_distance': float(euclidean_distances_arr[target_code_idx]),
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
            'model_type': 'DualCalciumVQVAE_Constrained',
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
    filename = "all_code_magnitudes_dual_constrained.json"
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
            'model_type': 'DualCalciumVQVAE_Constrained',
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
    
    print(f"\n✅ Tutte le ampiezze CONSTRAINED salvate in: {filepath}")
    
    return filepath


# ============================================================================
# FUNZIONI - VISUALIZZAZIONE E W&B
# ============================================================================

def plot_amplitude_vs_distance_dual(all_results):
    """
    Crea scatter plots: Distanza vs Differenza Ampiezza
    Separato per tipo di coppia (active-active, active-inactive, inactive-inactive)
    """
    
    # Raccogli tutti i dati
    all_pairs = []
    for result in all_results:
        # Carica i dati JSON per questo base code
        json_file = result['main_file']
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_pairs.extend(data['pairs'])
    
    # Separa per tipo di coppia
    pairs_aa = [p for p in all_pairs if p['base_codebook_type'] == 'active' and p['target_codebook_type'] == 'active']
    pairs_ai = [p for p in all_pairs if (p['base_codebook_type'] == 'active' and p['target_codebook_type'] == 'inactive') or 
                (p['base_codebook_type'] == 'inactive' and p['target_codebook_type'] == 'active')]
    pairs_ii = [p for p in all_pairs if p['base_codebook_type'] == 'inactive' and p['target_codebook_type'] == 'inactive']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Active-Active
    if pairs_aa:
        dists_aa = [p['cosine_distance'] for p in pairs_aa]
        amp_diffs_aa = [p['amplitude_difference_abs'] for p in pairs_aa]
        axes[0].scatter(dists_aa, amp_diffs_aa, alpha=0.5, s=20, color='red')
        axes[0].set_xlabel('Cosine Distance')
        axes[0].set_ylabel('Amplitude Difference (abs)')
        axes[0].set_title(f'Active-Active Pairs (n={len(pairs_aa)})', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        if len(dists_aa) > 1:
            corr_aa = np.corrcoef(dists_aa, amp_diffs_aa)[0, 1]
            axes[0].text(0.05, 0.95, f'Corr: {corr_aa:.3f}', transform=axes[0].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Active-Inactive
    if pairs_ai:
        dists_ai = [p['cosine_distance'] for p in pairs_ai]
        amp_diffs_ai = [p['amplitude_difference_abs'] for p in pairs_ai]
        axes[1].scatter(dists_ai, amp_diffs_ai, alpha=0.5, s=20, color='purple')
        axes[1].set_xlabel('Cosine Distance')
        axes[1].set_ylabel('Amplitude Difference (abs)')
        axes[1].set_title(f'Active-Inactive Pairs (n={len(pairs_ai)})', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        if len(dists_ai) > 1:
            corr_ai = np.corrcoef(dists_ai, amp_diffs_ai)[0, 1]
            axes[1].text(0.05, 0.95, f'Corr: {corr_ai:.3f}', transform=axes[1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Inactive-Inactive
    if pairs_ii:
        dists_ii = [p['cosine_distance'] for p in pairs_ii]
        amp_diffs_ii = [p['amplitude_difference_abs'] for p in pairs_ii]
        axes[2].scatter(dists_ii, amp_diffs_ii, alpha=0.5, s=20, color='blue')
        axes[2].set_xlabel('Cosine Distance')
        axes[2].set_ylabel('Amplitude Difference (abs)')
        axes[2].set_title(f'Inactive-Inactive Pairs (n={len(pairs_ii)})', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        if len(dists_ii) > 1:
            corr_ii = np.corrcoef(dists_ii, amp_diffs_ii)[0, 1]
            axes[2].text(0.05, 0.95, f'Corr: {corr_ii:.3f}', transform=axes[2].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('Amplitude Difference vs Cosine Distance (DUAL CONSTRAINED Codebook)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    global ALL_BASE_CODES
    
    print("="*70)
    print("📊 ANALISI AMPIEZZE DUAL CODEBOOK CONSTRAINED - JSON Export Version")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Num embeddings TOTAL: {MODEL_CONFIG['num_embeddings']}")
    print(f"   Num embeddings ACTIVE: {MODEL_CONFIG['num_embeddings']//2}")
    print(f"   Num embeddings INACTIVE: {MODEL_CONFIG['num_embeddings']//2}")
    print(f"   Base codes da analizzare: {len(ALL_BASE_CODES)} ({ALL_BASE_CODES[0]} to {ALL_BASE_CODES[-1]})")
    print(f"   Output directory: {OUTPUT_DIR}/")
    print(f"   W&B Project: {WANDB_PROJECT}")
    
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
            "num_base_codes": len(ALL_BASE_CODES),
            **MODEL_CONFIG
        },
        tags=["amplitude", "distance", "dual-codebook", "constrained", "codebook-analysis"]
    )
    
    print(f"✅ W&B inizializzato: {wandb.run.url}")
    
    try:
        # 1. Carica modello dual
        model, device = load_dual_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
        print(f"💻 Device finale: {device}")
        
        # 2. Estrai embeddings dual
        active_emb, inactive_emb, split_idx, all_emb = extract_dual_codebook_embeddings(model)
        num_codes = all_emb.shape[0]
        
        # Aggiorna ALL_BASE_CODES se necessario
        if len(ALL_BASE_CODES) != num_codes:
            ALL_BASE_CODES = list(range(num_codes))
            print(f"   ⚠️  ALL_BASE_CODES aggiornato a: {len(ALL_BASE_CODES)} codici")
        
        print(f"\n🔍 Totale codici nel codebook: {num_codes}")
        print(f"   Active: {len(active_emb)}")
        print(f"   Inactive: {len(inactive_emb)}")
        
        # 3. Calcola tutte le ampiezze
        print(f"\n{'='*70}")
        print("📏 CALCOLO AMPIEZZE (L2 NORM) - DUAL CODEBOOK CONSTRAINED")
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
        
        # 6. Crea visualizzazioni e salva su W&B
        print(f"\n{'='*70}")
        print("📊 CREAZIONE GRAFICI E SALVATAGGIO SU W&B")
        print(f"{'='*70}")
        
        fig = plot_amplitude_vs_distance_dual(all_results)
        wandb.log({"amplitude_vs_distance/dual_codebook_constrained": wandb.Image(fig)})
        plt.close(fig)
        print("   ✅ Grafico Amplitude vs Distance salvato su W&B")
        
        # Statistiche aggregate per W&B
        all_pairs = []
        for result in all_results:
            json_file = result['main_file']
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_pairs.extend(data['pairs'])
        
        pairs_aa = [p for p in all_pairs if p['base_codebook_type'] == 'active' and p['target_codebook_type'] == 'active']
        pairs_ai = [p for p in all_pairs if (p['base_codebook_type'] == 'active' and p['target_codebook_type'] == 'inactive') or 
                    (p['base_codebook_type'] == 'inactive' and p['target_codebook_type'] == 'active')]
        pairs_ii = [p for p in all_pairs if p['base_codebook_type'] == 'inactive' and p['target_codebook_type'] == 'inactive']
        
        if pairs_aa:
            dists_aa = [p['cosine_distance'] for p in pairs_aa]
            amp_diffs_aa = [p['amplitude_difference_abs'] for p in pairs_aa]
            if len(dists_aa) > 1:
                corr_aa = np.corrcoef(dists_aa, amp_diffs_aa)[0, 1]
                wandb.log({"correlations/active_active_amplitude_distance": corr_aa})
        
        if pairs_ai:
            dists_ai = [p['cosine_distance'] for p in pairs_ai]
            amp_diffs_ai = [p['amplitude_difference_abs'] for p in pairs_ai]
            if len(dists_ai) > 1:
                corr_ai = np.corrcoef(dists_ai, amp_diffs_ai)[0, 1]
                wandb.log({"correlations/active_inactive_amplitude_distance": corr_ai})
        
        if pairs_ii:
            dists_ii = [p['cosine_distance'] for p in pairs_ii]
            amp_diffs_ii = [p['amplitude_difference_abs'] for p in pairs_ii]
            if len(dists_ii) > 1:
                corr_ii = np.corrcoef(dists_ii, amp_diffs_ii)[0, 1]
                wandb.log({"correlations/inactive_inactive_amplitude_distance": corr_ii})
        
        # 7. Stampa riepilogo finale
        print(f"\n{'='*70}")
        print("✅ ANALISI COMPLETATA PER TUTTI I CODICI!")
        print(f"{'='*70}")
        print(f"\n📁 File salvati in: {OUTPUT_DIR}/coppie codice/")
        print(f"   - Totale base codes analizzati: {len(ALL_BASE_CODES)}")
        print(f"     • Active: {sum(1 for r in all_results if r['base_type'] == 'active')}")
        print(f"     • Inactive: {sum(1 for r in all_results if r['base_type'] == 'inactive')}")
        print(f"   - File JSON coppie: {len(all_results)}")
        print(f"   - File ampiezze individuali: {all_mags_file}")
        print(f"\n🌐 Tutti i risultati su W&B: {wandb.run.url}")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        wandb.finish()
        print("✅ W&B run completato!")


if __name__ == "__main__":
    main()

