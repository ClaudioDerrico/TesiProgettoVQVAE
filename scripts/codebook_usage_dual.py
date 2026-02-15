#!/usr/bin/env python3
"""
Analisi Utilizzo Codebook VQ-VAE - Dual Codebook

Analizza quali codici dei DUE codebook vengono effettivamente utilizzati
durante l'encoding di un dataset per modelli DUAL CODEBOOK.

Uso:
    python scripts/codebook_usage_dual.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from collections import Counter

from models.dual_vqvae import DualCalciumVQVAE
from datasets.calcium import (
    UnifiedMultiSessionDataset,
    TRAINING_SESSION_IDS,
    TEST_SESSION_IDS
)

# ============================================================================
# ⚙️ CONFIGURAZIONE - MODIFICA QUI PER CAMBIARE CHECKPOINT E PARAMETRI
# ============================================================================

# ✅ MODIFICA QUI: Inserisci il path al tuo checkpoint dual
# Esempi:
#   "results/dual_codebook/best_dual_model_32_constrained.pth"
#   "results/dual_codebook/best_dual_model_1024_constrained.pth"
CHECKPOINT_PATH = "results/dual_codebook/best_dual_model_2048_constrained.pth"

# ✅ MODIFICA QUI: Configurazione modello (deve corrispondere al checkpoint)
MODEL_CONFIG = {
    "num_neurons": 1,
    "num_hiddens": 128,
    
    "num_residual_layers": 3,
    "num_residual_hiddens": 64,
    "num_embeddings": 2048,        # ⬅️ CAMBIA: 16, 32, 64, 128, 256, 512, 1024, etc.
    "embedding_dim": 128,         # ⬅️ CAMBIA: tipicamente = num_embeddings per piccoli, 64 per grandi
    "commitment_cost": 0.25,
    "dropout_rate": 0.1,
    "use_quantizer": True,
    'threshold': 0.0,
    'threshold_type': 'adaptive',
}

# Parametri analisi
DATASET_TO_ANALYZE = 'train'  # 'train', 'test', o 'both'
MAX_SAMPLES = None  # None = tutti i campioni
BATCH_SIZE = 64
SAVE_DIR = 'code_usage'  # Salva nella stessa cartella dei single
USE_WANDB = False

# ============================================================================
# FUNZIONI
# ============================================================================

def load_model(checkpoint_path, config, device):
    """Carica il modello dual codebook dai pesi salvati"""
    print(f"🔄 Caricando modello DUAL da: {checkpoint_path}")
    
    # Carica checkpoint per leggere configurazione
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # ✅ Prova a leggere configurazione dal checkpoint
    if 'model_config' in checkpoint:
        checkpoint_config = checkpoint['model_config']
        print(f"📋 Configurazione trovata nel checkpoint:")
        print(f"   num_embeddings: {checkpoint_config.get('num_embeddings', 'N/A')}")
        # Usa configurazione dal checkpoint, con fallback su config fornito
        final_config = {
            'num_neurons': checkpoint_config.get('num_neurons', config['num_neurons']),
            'num_hiddens': checkpoint_config.get('num_hiddens', config['num_hiddens']),
            'num_residual_layers': checkpoint_config.get('num_residual_layers', config['num_residual_layers']),
            'num_residual_hiddens': checkpoint_config.get('num_residual_hiddens', config['num_residual_hiddens']),
            'num_embeddings': checkpoint_config.get('num_embeddings', config['num_embeddings']),
            'embedding_dim': checkpoint_config.get('embedding_dim', config['embedding_dim']),
            'commitment_cost': checkpoint_config.get('commitment_cost', config['commitment_cost']),
            'dropout_rate': checkpoint_config.get('dropout_rate', config['dropout_rate']),
            'use_quantizer': checkpoint_config.get('use_quantizer', config['use_quantizer']),
            'threshold': checkpoint_config.get('threshold', config.get('threshold', 0.0)),
            'threshold_type': checkpoint_config.get('threshold_type', config.get('threshold_type', 'adaptive'))
        }
    else:
        # ✅ Estrai num_embeddings dalle dimensioni dei pesi se disponibile
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Prova a estrarre num_embeddings dalle dimensioni
        if 'vector_quantization.embedding_active.weight' in state_dict:
            embedding_active_shape = state_dict['vector_quantization.embedding_active.weight'].shape
            # Per dual codebook: num_embeddings_total = embedding_active.shape[0] * 2
            num_embeddings_from_checkpoint = embedding_active_shape[0] * 2
            print(f"📋 Num embeddings estratto dal checkpoint: {num_embeddings_from_checkpoint}")
            final_config = config.copy()
            final_config['num_embeddings'] = num_embeddings_from_checkpoint
        else:
            print(f"⚠️  Configurazione non trovata nel checkpoint, uso configurazione fornita")
            final_config = config
    
    # Crea modello con configurazione corretta
    model = DualCalciumVQVAE(
        num_neurons=final_config['num_neurons'],
        num_hiddens=final_config['num_hiddens'],
        num_residual_layers=final_config['num_residual_layers'],
        num_residual_hiddens=final_config['num_residual_hiddens'],
        num_embeddings=final_config['num_embeddings'],
        embedding_dim=final_config['embedding_dim'],
        commitment_cost=final_config['commitment_cost'],
        dropout_rate=final_config['dropout_rate'],
        use_quantizer=final_config['use_quantizer'],
        threshold=final_config.get('threshold', 0.0),
        threshold_type=final_config.get('threshold_type', 'adaptive')
    )
    
    # Carica pesi
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Modello caricato (epoca {checkpoint.get('epoch', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Modello caricato")
    
    model = model.to(device)
    model.eval()
    
    # Info codebook
    n_e_per_codebook = model.vector_quantization.n_e_per_codebook
    print(f"📚 Codebook Active: {n_e_per_codebook} codici")
    print(f"📚 Codebook Inactive: {n_e_per_codebook} codici")
    print(f"📚 Total: {model.vector_quantization.n_e} codici")
    
    return model


def analyze_dual_codebook_usage(model, dataloader, device, max_samples=None):
    """
    Analizza l'utilizzo dei DUE codebook su un dataset.
    
    Returns:
        dict con statistiche di utilizzo per entrambi i codebook
    """
    print(f"\n{'='*70}")
    print(f"📊 ANALISI UTILIZZO DUAL CODEBOOK")
    print(f"{'='*70}")
    
    n_e_total = model.vector_quantization.n_e
    n_e_per_codebook = model.vector_quantization.n_e_per_codebook
    
    print(f"   Total codebook size: {n_e_total} codici")
    print(f"   Active codebook: {n_e_per_codebook} codici (0-{n_e_per_codebook-1})")
    print(f"   Inactive codebook: {n_e_per_codebook} codici ({n_e_per_codebook}-{n_e_total-1})")
    
    # Counter per entrambi i codebook
    code_usage_active = Counter()
    code_usage_inactive = Counter()
    total_encodings = 0
    total_active = 0
    total_inactive = 0
    
    model.eval()
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 3:
                neural_data = batch_data[0].to(device)
            else:
                neural_data = batch_data.to(device)
            
            # ✅ Assicurati che i dati abbiano 1 neurone (come il modello dual single neuron)
            # Il modello dual è stato addestrato con num_neurons=1
            if neural_data.dim() == 3 and neural_data.shape[1] > 1:
                # Se abbiamo più neuroni, prendi solo il primo
                neural_data = neural_data[:, 0:1, :]  # (batch, 1, 60)
            elif neural_data.dim() == 2:
                # Se è (batch, 60), aggiungi dimensione neurone
                neural_data = neural_data.unsqueeze(1)  # (batch, 1, 60)
            
            # Encode
            z = model.encoder(neural_data)
            z = model.pre_quantization_conv(z)
            
            # Quantize e ottieni gli indici
            _, _, _, _, encoding_indices = model.vector_quantization(z)
            
            # Separa indici active e inactive
            indices_cpu = encoding_indices.cpu().numpy().flatten()
            
            # Active: indici < n_e_per_codebook
            active_mask = indices_cpu < n_e_per_codebook
            active_indices = indices_cpu[active_mask]
            code_usage_active.update(active_indices)
            total_active += len(active_indices)
            
            # Inactive: indici >= n_e_per_codebook (ma convertiamo a 0-based per inactive codebook)
            inactive_mask = indices_cpu >= n_e_per_codebook
            inactive_indices = indices_cpu[inactive_mask] - n_e_per_codebook  # Converti a 0-based
            code_usage_inactive.update(inactive_indices)
            total_inactive += len(inactive_indices)
            
            total_encodings += len(indices_cpu)
            samples_processed += neural_data.shape[0]
            
            if (batch_idx + 1) % 50 == 0:
                print(f"   Processati {batch_idx + 1} batch, {samples_processed} campioni...")
            
            if max_samples and samples_processed >= max_samples:
                break
    
    # Calcola statistiche per ACTIVE codebook
    used_codes_active = len(code_usage_active)
    usage_percentage_active = (used_codes_active / n_e_per_codebook) * 100
    
    code_frequencies_active = {}
    for code_idx in range(n_e_per_codebook):
        count = code_usage_active.get(code_idx, 0)
        code_frequencies_active[code_idx] = {
            'count': count,
            'percentage': (count / total_active * 100) if total_active > 0 else 0
        }
    
    unused_codes_active = [i for i in range(n_e_per_codebook) if code_usage_active[i] == 0]
    top_10_codes_active = code_usage_active.most_common(10)
    
    # Calcola statistiche per INACTIVE codebook
    used_codes_inactive = len(code_usage_inactive)
    usage_percentage_inactive = (used_codes_inactive / n_e_per_codebook) * 100
    
    code_frequencies_inactive = {}
    for code_idx in range(n_e_per_codebook):
        count = code_usage_inactive.get(code_idx, 0)
        code_frequencies_inactive[code_idx] = {
            'count': count,
            'percentage': (count / total_inactive * 100) if total_inactive > 0 else 0
        }
    
    unused_codes_inactive = [i for i in range(n_e_per_codebook) if code_usage_inactive[i] == 0]
    top_10_codes_inactive = code_usage_inactive.most_common(10)
    
    # Statistiche combinate
    total_used = used_codes_active + used_codes_inactive
    total_usage_percentage = (total_used / n_e_total) * 100
    
    print(f"\n✅ Analisi completata:")
    print(f"   Campioni processati: {samples_processed}")
    print(f"   Total encodings: {total_encodings:,}")
    print(f"   Active encodings: {total_active:,} ({total_active/total_encodings*100:.1f}%)")
    print(f"   Inactive encodings: {total_inactive:,} ({total_inactive/total_encodings*100:.1f}%)")
    print(f"\n📊 ACTIVE Codebook:")
    print(f"   Codici usati: {used_codes_active}/{n_e_per_codebook}")
    print(f"   Utilizzo: {usage_percentage_active:.2f}%")
    print(f"   Codici inutilizzati: {len(unused_codes_active)}")
    print(f"\n📊 INACTIVE Codebook:")
    print(f"   Codici usati: {used_codes_inactive}/{n_e_per_codebook}")
    print(f"   Utilizzo: {usage_percentage_inactive:.2f}%")
    print(f"   Codici inutilizzati: {len(unused_codes_inactive)}")
    print(f"\n📊 COMBINED:")
    print(f"   Total codici usati: {total_used}/{n_e_total}")
    print(f"   Total utilizzo: {total_usage_percentage:.2f}%")
    
    return {
        'num_embeddings_total': n_e_total,
        'num_embeddings_per_codebook': n_e_per_codebook,
        'active': {
            'num_embeddings': n_e_per_codebook,
            'used_codes': used_codes_active,
            'usage_percentage': usage_percentage_active,
            'unused_codes': unused_codes_active,
            'code_frequencies': code_frequencies_active,
            'top_10_codes': top_10_codes_active,
            'total_encodings': total_active,
            'code_usage_counter': code_usage_active
        },
        'inactive': {
            'num_embeddings': n_e_per_codebook,
            'used_codes': used_codes_inactive,
            'usage_percentage': usage_percentage_inactive,
            'unused_codes': unused_codes_inactive,
            'code_frequencies': code_frequencies_inactive,
            'top_10_codes': top_10_codes_inactive,
            'total_encodings': total_inactive,
            'code_usage_counter': code_usage_inactive
        },
        'combined': {
            'used_codes': total_used,
            'usage_percentage': total_usage_percentage,
            'total_encodings': total_encodings,
            'samples_processed': samples_processed
        }
    }


def save_results_to_json(results, checkpoint_path, model_config, dataset_type):
    """Salva i risultati in formato JSON"""
    output_dir = Path(SAVE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_data = {
        'metadata': {
            'checkpoint_path': str(checkpoint_path),
            'dataset_analyzed': dataset_type,
            'model_config': model_config,
            'model_type': 'DualCalciumVQVAE'
        },
        'results': {}
    }
    
    for dataset_name, stats in results.items():
        # Active codebook
        active_stats = stats['active']
        code_usage_active_dict = {int(k): int(v) for k, v in active_stats['code_usage_counter'].items()}
        top_10_active = [
            {'code_index': int(code_idx), 'count': int(count)}
            for code_idx, count in active_stats['top_10_codes']
        ]
        code_frequencies_active = {
            str(k): {
                'count': int(v['count']),
                'percentage': float(v['percentage'])
            }
            for k, v in active_stats['code_frequencies'].items()
        }
        
        # Inactive codebook
        inactive_stats = stats['inactive']
        code_usage_inactive_dict = {int(k): int(v) for k, v in inactive_stats['code_usage_counter'].items()}
        top_10_inactive = [
            {'code_index': int(code_idx), 'count': int(count)}
            for code_idx, count in inactive_stats['top_10_codes']
        ]
        code_frequencies_inactive = {
            str(k): {
                'count': int(v['count']),
                'percentage': float(v['percentage'])
            }
            for k, v in inactive_stats['code_frequencies'].items()
        }
        
        json_data['results'][dataset_name] = {
            'num_embeddings_total': int(stats['num_embeddings_total']),
            'num_embeddings_per_codebook': int(stats['num_embeddings_per_codebook']),
            'active': {
                'num_embeddings': int(active_stats['num_embeddings']),
                'used_codes': int(active_stats['used_codes']),
                'usage_percentage': float(active_stats['usage_percentage']),
                'unused_codes': [int(x) for x in active_stats['unused_codes']],
                'total_encodings': int(active_stats['total_encodings']),
                'code_usage': code_usage_active_dict,
                'top_10_codes': top_10_active,
                'code_frequencies': code_frequencies_active
            },
            'inactive': {
                'num_embeddings': int(inactive_stats['num_embeddings']),
                'used_codes': int(inactive_stats['used_codes']),
                'usage_percentage': float(inactive_stats['usage_percentage']),
                'unused_codes': [int(x) for x in inactive_stats['unused_codes']],
                'total_encodings': int(inactive_stats['total_encodings']),
                'code_usage': code_usage_inactive_dict,
                'top_10_codes': top_10_inactive,
                'code_frequencies': code_frequencies_inactive
            },
            'combined': {
                'used_codes': int(stats['combined']['used_codes']),
                'usage_percentage': float(stats['combined']['usage_percentage']),
                'total_encodings': int(stats['combined']['total_encodings']),
                'samples_processed': int(stats['combined']['samples_processed'])
            }
        }
    
    checkpoint_name = Path(checkpoint_path).stem
    filename = f'{checkpoint_name}.json'
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Risultati salvati in JSON:")
    print(f"   {output_path}")
    
    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Carica modello
    model = load_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
    
    # Carica dataset (single neuron come il modello dual)
    print(f"\n{'='*70}")
    print(f"📊 CARICAMENTO DATASET")
    print(f"{'='*70}")
    
    window_size = 60
    stride = 50
    num_neurons = MODEL_CONFIG['num_neurons']  # 1 per single neuron
    
    if DATASET_TO_ANALYZE in ['train', 'both']:
        train_dataset = UnifiedMultiSessionDataset(
            session_ids=TRAINING_SESSION_IDS,
            window_size=window_size,
            stride=stride,
            min_neurons=num_neurons,
            mode='train'
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        print(f"✅ Train dataset caricato: {len(train_dataset)} campioni")
        train_results = analyze_dual_codebook_usage(model, train_dataloader, device, MAX_SAMPLES)
        results = {'train': train_results}
    
    if DATASET_TO_ANALYZE == 'both':
        # Per test, usa TEST_SESSION_IDS
        test_dataset = UnifiedMultiSessionDataset(
            session_ids=TEST_SESSION_IDS,
            window_size=window_size,
            stride=stride,
            min_neurons=num_neurons,
            mode='test'
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        print(f"✅ Test dataset caricato: {len(test_dataset)} campioni")
        test_results = analyze_dual_codebook_usage(model, test_dataloader, device, MAX_SAMPLES)
        results['test'] = test_results
    
    # Salva JSON
    save_results_to_json(results, CHECKPOINT_PATH, MODEL_CONFIG, DATASET_TO_ANALYZE)
    
    print("\n" + "="*70)
    print("✅ ANALISI COMPLETATA!")
    print("="*70)


if __name__ == "__main__":
    main()

