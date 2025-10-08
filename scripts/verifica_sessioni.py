#!/usr/bin/env python3
"""
Script di verifica CORRETTO che tiene conto del padding circolare.

Verifica il rapporto campioni/area distinguendo tra:
- TRAINING: Seleziona BEST 30 neuroni (fissi)
- TEST: Sliding su TUTTI i neuroni con padding circolare (nessuna finestra scartata)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datasets.calcium import (
    SimpleAllenBrainDataset,
    TRAINING_SESSION_IDS,
    TEST_SESSION_IDS,
    extract_speed_from_hdf5,
    preprocess_neural_data_only
)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

WINDOW_SIZE = 60
STRIDE = 50
MIN_NEURONS = 30
NEURON_STRIDE = 30

# ============================================================================
# FUNZIONI HELPER
# ============================================================================

def analyze_session(session_id, window_size, stride, min_neurons, 
                   is_training=True, neuron_stride=30):
    """
    Analizza sessione con modalità corretta e calcolo CORRETTO dei gruppi neuroni.
    
    TRAINING: use_neuron_sliding=False (BEST 30 neuroni)
    TEST: use_neuron_sliding=True (ALL neuroni con padding circolare)
    """
    
    mode = "TRAINING" if is_training else "TEST"
    print(f"\n{'='*70}")
    print(f"📊 Analizzando sessione {mode}: {session_id}")
    print(f"{'='*70}")
    
    try:
        # 1. Carica dati RAW
        print("🔍 Step 1: Caricamento dati RAW...")
        extraction_result = extract_speed_from_hdf5(session_id)
        
        if extraction_result is None or extraction_result.get('dff_data') is None:
            print(f"❌ Impossibile caricare dati")
            return None
        
        processed_data = preprocess_neural_data_only(extraction_result)
        if processed_data is None:
            print(f"❌ Preprocessing fallito")
            return None
        
        dff_traces = processed_data['dff_traces']
        if dff_traces.ndim == 2 and dff_traces.shape[0] > dff_traces.shape[1]:
            dff_traces = dff_traces.T
        
        total_neurons_raw = dff_traces.shape[0]
        total_timepoints_raw = dff_traces.shape[1]
        
        print(f"✅ Dati RAW:")
        print(f"   Total neurons: {total_neurons_raw}")
        print(f"   Total timepoints: {total_timepoints_raw}")
        
        # 2. Crea dataset con modalità corretta
        print(f"\n🔍 Step 2: Creazione dataset ({mode} mode)...")
        
        if is_training:
            print(f"   🔴 TRAINING: Selecting BEST {min_neurons} neurons")
            dataset = SimpleAllenBrainDataset(
                window_size=window_size,
                stride=stride,
                min_neurons=min_neurons,
                session_id=session_id,
                use_neuron_sliding=False,  # ← TRAINING: fisso
                neuron_selection_method='combined'
            )
        else:
            print(f"   🔵 TEST: Sliding on ALL neurons (stride={neuron_stride}) with circular padding")
            dataset = SimpleAllenBrainDataset(
                window_size=window_size,
                stride=stride,
                min_neurons=min_neurons,
                session_id=session_id,
                neuron_stride=neuron_stride,
                use_neuron_sliding=True  # ← TEST: sliding
            )
        
        num_samples = len(dataset)
        window_area = min_neurons * window_size
        
        print(f"✅ Dataset creato:")
        print(f"   Samples: {num_samples}")
        print(f"   Window area: {window_area}")
        
        # 3. Calcola statistiche specifiche per modalità
        if is_training:
            # ========================================
            # TRAINING MODE
            # ========================================
            expected_temporal_windows = (total_timepoints_raw - window_size) // stride + 1
            remainder_time = (total_timepoints_raw - window_size) % stride
            if remainder_time != 0:
                expected_temporal_windows += 1
            
            ratio = num_samples / window_area
            discrepancy = abs(num_samples - expected_temporal_windows)
            
            print(f"\n📐 TRAINING Statistics:")
            print(f"   Expected temporal windows: {expected_temporal_windows}")
            print(f"   Actual samples: {num_samples}")
            print(f"   Discrepancy: {discrepancy}")
            print(f"   Ratio (samples/area): {ratio:.4f}")
            
            results = {
                'session_id': session_id,
                'mode': 'training',
                'success': True,
                'total_neurons_raw': total_neurons_raw,
                'neurons_used': min_neurons,
                'total_timepoints_raw': total_timepoints_raw,
                'num_samples': num_samples,
                'window_area': window_area,
                'ratio': ratio,
                'expected_temporal_windows': expected_temporal_windows,
                'discrepancy': discrepancy,
                'neuron_groups': 1
            }
            
        else:
            # ========================================
            # TEST MODE (con padding circolare)
            # ========================================
            
            # Finestre temporali
            expected_temporal_windows = (total_timepoints_raw - window_size) // stride + 1
            remainder_time = (total_timepoints_raw - window_size) % stride
            if remainder_time != 0:
                expected_temporal_windows += 1
            
            # ✅ GRUPPI NEURONI con PADDING CIRCOLARE (ceil division)
            # Formula: ceil(total_neurons / neuron_stride)
            num_neuron_groups = (total_neurons_raw + neuron_stride - 1) // neuron_stride
            
            # Calcola quanti neuroni servono per il padding nell'ultimo gruppo
            neurons_in_last_group = total_neurons_raw - (num_neuron_groups - 1) * neuron_stride
            
            if neurons_in_last_group < min_neurons:
                # L'ultimo gruppo ha padding circolare
                padding_neurons_last_group = min_neurons - neurons_in_last_group
                has_padding = True
            else:
                padding_neurons_last_group = 0
                has_padding = False
            
            # Samples totali attesi
            expected_total_samples = expected_temporal_windows * num_neuron_groups
            
            ratio = num_samples / window_area
            discrepancy = abs(num_samples - expected_total_samples)
            
            print(f"\n📐 TEST Statistics:")
            print(f"   Temporal windows: {expected_temporal_windows}")
            print(f"   Neuron groups (with padding): {num_neuron_groups}")
            print(f"   Expected total samples: {expected_temporal_windows} × {num_neuron_groups} = {expected_total_samples}")
            print(f"   Actual samples: {num_samples}")
            print(f"   Discrepancy: {discrepancy}")
            print(f"   Ratio (samples/area): {ratio:.4f}")
            
            # Dettaglio gruppi neuroni
            print(f"\n   🔍 Neuron groups detail:")
            for g in range(num_neuron_groups):
                start_n = g * neuron_stride
                end_n = min(start_n + min_neurons, total_neurons_raw)
                
                if g == num_neuron_groups - 1 and has_padding:
                    # Ultimo gruppo con padding
                    neurons_real = neurons_in_last_group
                    neurons_padded = padding_neurons_last_group
                    padding_start = 0
                    padding_end = neurons_padded
                    
                    print(f"      Group {g+1}: neurons [{start_n}:{end_n}] + circular [{padding_start}:{padding_end}]")
                    print(f"               ({neurons_real} real + {neurons_padded} padded)")
                else:
                    # Gruppo completo
                    print(f"      Group {g+1}: neurons [{start_n}:{end_n}] (all real)")
            
            results = {
                'session_id': session_id,
                'mode': 'test',
                'success': True,
                'total_neurons_raw': total_neurons_raw,
                'neurons_used': min_neurons,
                'total_timepoints_raw': total_timepoints_raw,
                'num_samples': num_samples,
                'window_area': window_area,
                'ratio': ratio,
                'expected_temporal_windows': expected_temporal_windows,
                'num_neuron_groups': num_neuron_groups,
                'expected_total_samples': expected_total_samples,
                'discrepancy': discrepancy,
                'has_padding': has_padding,
                'padding_neurons_last_group': padding_neurons_last_group
            }
        
        return results
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        return {'session_id': session_id, 'success': False, 'error': str(e)}


def print_comparison_table(train_results, test_results):
    """Tabella comparativa train vs test"""
    
    print("\n" + "="*110)
    print("📊 CONFRONTO TRAIN vs TEST: Campioni Generati (CON PADDING CIRCOLARE)")
    print("="*110)
    
    # TRAINING
    print("\n🔴 TRAINING SESSIONS (BEST 30 neurons per session):")
    print("-"*110)
    
    header = f"{'Session':<12} | {'Neurons':<8} | {'Used':<5} | {'Timepoints':<11} | {'Samples':<8} | {'Groups':<7} | {'Ratio':<10}"
    print(header)
    print("-"*110)
    
    for r in train_results:
        if r and r.get('success'):
            row = (f"{r['session_id']:<12} | "
                   f"{r['total_neurons_raw']:<8} | "
                   f"{r['neurons_used']:<5} | "
                   f"{r['total_timepoints_raw']:<11} | "
                   f"{r['num_samples']:<8} | "
                   f"{r['neuron_groups']:<7} | "
                   f"{r['ratio']:<10.4f}")
            print(row)
    
    # TEST
    print("\n🔵 TEST SESSIONS (ALL neurons with sliding + circular padding):")
    print("-"*110)
    
    header = f"{'Session':<12} | {'Neurons':<8} | {'Used':<5} | {'Timepoints':<11} | {'Samples':<8} | {'Groups':<7} | {'Padding':<8} | {'Ratio':<10}"
    print(header)
    print("-"*110)
    
    for r in test_results:
        if r and r.get('success'):
            padding_str = f"Yes ({r.get('padding_neurons_last_group', 0)})" if r.get('has_padding') else "No"
            row = (f"{r['session_id']:<12} | "
                   f"{r['total_neurons_raw']:<8} | "
                   f"{r['neurons_used']:<5} | "
                   f"{r['total_timepoints_raw']:<11} | "
                   f"{r['num_samples']:<8} | "
                   f"{r['num_neuron_groups']:<7} | "
                   f"{padding_str:<8} | "
                   f"{r['ratio']:<10.4f}")
            print(row)
    
    print("="*110)
    
    # Statistiche aggregate
    train_samples = [r['num_samples'] for r in train_results if r.get('success')]
    test_samples = [r['num_samples'] for r in test_results if r.get('success')]
    
    train_ratios = [r['ratio'] for r in train_results if r.get('success')]
    test_ratios = [r['ratio'] for r in test_results if r.get('success')]
    
    test_groups = [r['num_neuron_groups'] for r in test_results if r.get('success')]
    test_padded = sum(1 for r in test_results if r.get('success') and r.get('has_padding'))
    
    print(f"\n📊 STATISTICHE AGGREGATE:")
    print(f"\n   TRAINING (n={len(train_samples)}):")
    print(f"      Samples mean: {np.mean(train_samples):.0f} ± {np.std(train_samples):.0f}")
    print(f"      Ratio mean: {np.mean(train_ratios):.4f} ± {np.std(train_ratios):.4f}")
    
    print(f"\n   TEST (n={len(test_samples)}):")
    print(f"      Samples mean: {np.mean(test_samples):.0f} ± {np.std(test_samples):.0f}")
    print(f"      Ratio mean: {np.mean(test_ratios):.4f} ± {np.std(test_ratios):.4f}")
    print(f"      Groups mean: {np.mean(test_groups):.1f} ± {np.std(test_groups):.1f}")
    print(f"      Sessions with padding: {test_padded}/{len(test_results)}")
    
    # Moltiplicatore medio
    if train_samples and test_samples:
        multiplier = np.mean(test_samples) / np.mean(train_samples)
        print(f"\n   🔍 TEST/TRAIN ratio: {multiplier:.2f}×")
        print(f"       → Test ha in media {multiplier:.1f}× più campioni del train")
        print(f"       → Questo perché il test usa TUTTI i neuroni (con padding)")


def main():
    print("="*70)
    print("🔬 VERIFICA TRAIN (BEST 30) vs TEST (ALL neurons + padding)")
    print("="*70)
    print(f"\n📋 Configurazione:")
    print(f"   Window size: {WINDOW_SIZE}")
    print(f"   Stride: {STRIDE}")
    print(f"   Min neurons: {MIN_NEURONS}")
    print(f"   Neuron stride (test): {NEURON_STRIDE}")
    print(f"   ✅ Padding circolare: ABILITATO (nessuna finestra scartata)")
    
    # Analizza TRAINING sessions
    print("\n" + "="*70)
    print(f"🔴 TRAINING SESSIONS ({len(TRAINING_SESSION_IDS)} sessions)")
    print("="*70)
    
    train_results = []
    for session_id in TRAINING_SESSION_IDS:
        result = analyze_session(
            session_id, 
            WINDOW_SIZE, 
            STRIDE, 
            MIN_NEURONS,
            is_training=True
        )
        if result:
            train_results.append(result)
    
    # Analizza TEST sessions
    print("\n" + "="*70)
    print(f"🔵 TEST SESSIONS ({len(TEST_SESSION_IDS)} sessions)")
    print("="*70)
    
    test_results = []
    for session_id in TEST_SESSION_IDS:
        result = analyze_session(
            session_id, 
            WINDOW_SIZE, 
            STRIDE, 
            MIN_NEURONS,
            is_training=False,
            neuron_stride=NEURON_STRIDE
        )
        if result:
            test_results.append(result)
    
    # Confronto
    print_comparison_table(train_results, test_results)
    
    # Salva risultati
    import json
    from pathlib import Path
    
    save_dir = Path("verification_results")
    save_dir.mkdir(exist_ok=True)
    
    output = {
        'config': {
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'min_neurons': MIN_NEURONS,
            'neuron_stride': NEURON_STRIDE,
            'window_area': MIN_NEURONS * WINDOW_SIZE,
            'circular_padding_enabled': True
        },
        'training_sessions': train_results,
        'test_sessions': test_results
    }
    
    save_path = save_dir / "train_vs_test_with_padding.json"
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x)
    
    print(f"\n💾 Risultati salvati in: {save_path}")
    print("\n" + "="*70)
    print("✅ VERIFICA COMPLETATA!")
    print("="*70)


if __name__ == "__main__":
    main()