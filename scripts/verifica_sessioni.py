#!/usr/bin/env python3
"""
Script di verifica CORRETTO che tiene conto del padding circolare.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
   
    
from pathlib import Path
print("✅ Path importato")  # ← TEST 4
    
from datasets.calcium import (
    SimpleAllenBrainDataset,
    TRAINING_SESSION_IDS,
    TEST_SESSION_IDS,
    extract_speed_from_hdf5,
    preprocess_neural_data_only
)
print("✅ datasets.calcium importato")  # ← TEST 5

# CONFIGURAZIONE
WINDOW_SIZE = 60
STRIDE = 10
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


# ============================================================================
# NUOVE FUNZIONI PER ANALISI AVANZATA
# ============================================================================

def compare_session_lengths(train_results, test_results):
    """
    Confronta la lunghezza (timepoints totali) delle sessioni train vs test.
    """
    print("\n" + "="*70)
    print("📏 CONFRONTO LUNGHEZZA SESSIONI (Timepoints Totali)")
    print("="*70)
    
    # Estrai timepoints
    train_timepoints = [r['total_timepoints_raw'] for r in train_results if r.get('success')]
    test_timepoints = [r['total_timepoints_raw'] for r in test_results if r.get('success')]
    
    train_neurons = [r['total_neurons_raw'] for r in train_results if r.get('success')]
    test_neurons = [r['total_neurons_raw'] for r in test_results if r.get('success')]
    
    # Statistiche TRAINING
    print("\n🔴 TRAINING SESSIONS:")
    print(f"   Timepoints: {np.min(train_timepoints)} - {np.max(train_timepoints)}")
    print(f"   Mean: {np.mean(train_timepoints):.0f} ± {np.std(train_timepoints):.0f}")
    print(f"   Total neurons: {np.min(train_neurons)} - {np.max(train_neurons)}")
    print(f"   Mean neurons: {np.mean(train_neurons):.0f} ± {np.std(train_neurons):.0f}")
    
    # Statistiche TEST
    print("\n🔵 TEST SESSIONS:")
    print(f"   Timepoints: {np.min(test_timepoints)} - {np.max(test_timepoints)}")
    print(f"   Mean: {np.mean(test_timepoints):.0f} ± {np.std(test_timepoints):.0f}")
    print(f"   Total neurons: {np.min(test_neurons)} - {np.max(test_neurons)}")
    print(f"   Mean neurons: {np.mean(test_neurons):.0f} ± {np.std(test_neurons):.0f}")
    
    # Confronto
    ratio_timepoints = np.mean(test_timepoints) / np.mean(train_timepoints)
    ratio_neurons = np.mean(test_neurons) / np.mean(train_neurons)
    
    print("\n🔍 RAPPORTO TEST/TRAIN:")
    print(f"   Timepoints: {ratio_timepoints:.2f}× ({'più lunghe' if ratio_timepoints > 1 else 'più corte'})")
    print(f"   Neurons:    {ratio_neurons:.2f}× ({'più neuroni' if ratio_neurons > 1 else 'meno neuroni'})")
    
    # Calcola durata in secondi (assumendo ~30 Hz sampling rate)
    sampling_rate = 30  # Hz (tipico per calcium imaging)
    
    train_duration_sec = np.array(train_timepoints) / sampling_rate
    test_duration_sec = np.array(test_timepoints) / sampling_rate
    
    print(f"\n⏱️  DURATA STIMATA (@ {sampling_rate} Hz):")
    print(f"   TRAIN: {np.mean(train_duration_sec):.0f}s ({np.mean(train_duration_sec)/60:.1f}min)")
    print(f"   TEST:  {np.mean(test_duration_sec):.0f}s ({np.mean(test_duration_sec)/60:.1f}min)")
    
    print("="*70)


def compare_train_vs_test_coverage(train_results, test_results):
    """
    Confronta la copertura (samples/area) e altre metriche tra TRAIN e TEST.
    """
    print("\n" + "="*70)
    print("🎯 CONFRONTO TRAIN vs TEST: Coverage & Metriche")
    print("="*70)
    
    # Estrai metriche TRAINING
    train_samples = [r['num_samples'] for r in train_results if r.get('success')]
    train_ratios = [r['ratio'] for r in train_results if r.get('success')]
    train_neurons_used = [r['neurons_used'] for r in train_results if r.get('success')]
    
    # Estrai metriche TEST
    test_samples = [r['num_samples'] for r in test_results if r.get('success')]
    test_ratios = [r['ratio'] for r in test_results if r.get('success')]
    test_neurons_groups = [r['num_neuron_groups'] for r in test_results if r.get('success')]
    test_has_padding = [r.get('has_padding', False) for r in test_results if r.get('success')]
    
    # ========================================
    # TABELLA COMPARATIVA
    # ========================================
    print("\n📊 METRICHE AGGREGATE:")
    print("-"*70)
    
    header = f"{'Metrica':<30} | {'TRAIN':<18} | {'TEST':<18}"
    print(header)
    print("-"*70)
    
    # Numero sessioni
    row = f"{'Sessioni':<30} | {len(train_results):<18} | {len(test_results):<18}"
    print(row)
    
    # Samples totali
    row = f"{'Samples (totali)':<30} | {sum(train_samples):<18} | {sum(test_samples):<18}"
    print(row)
    
    # Samples medi per sessione
    row = f"{'Samples (mean/session)':<30} | {np.mean(train_samples):<18.0f} | {np.mean(test_samples):<18.0f}"
    print(row)
    
    # Ratio medio
    row = f"{'Ratio (samples/area) mean':<30} | {np.mean(train_ratios):<18.4f} | {np.mean(test_ratios):<18.4f}"
    print(row)
    
    # Neuroni usati
    row = f"{'Neurons used':<30} | {train_neurons_used[0]:<18} | {train_neurons_used[0]:<18}"
    print(row)
    
    # Gruppi neuroni (solo test)
    if test_neurons_groups:
        row = f"{'Neuron groups (mean)':<30} | {'N/A':<18} | {np.mean(test_neurons_groups):<18.1f}"
        print(row)
    
    # Padding (solo test)
    num_with_padding = sum(test_has_padding)
    row = f"{'Sessions with padding':<30} | {'N/A':<18} | {num_with_padding}/{len(test_results):<14}"
    print(row)
    
    print("-"*70)
    
    # ========================================
    # ANALISI COVERAGE
    # ========================================
    print("\n🔍 ANALISI COVERAGE:")
    
    # Calcola "data points" totali (neuroni × timesteps)
    train_total_datapoints = []
    test_total_datapoints = []
    
    for r in train_results:
        if r.get('success'):
            # TRAIN: usa solo BEST 30 neuroni
            datapoints = r['neurons_used'] * r['total_timepoints_raw']
            train_total_datapoints.append(datapoints)
    
    for r in test_results:
        if r.get('success'):
            # TEST: usa TUTTI i neuroni
            datapoints = r['total_neurons_raw'] * r['total_timepoints_raw']
            test_total_datapoints.append(datapoints)
    
    train_coverage = []
    for i, r in enumerate(train_results):
        if r.get('success'):
            # Coverage = (samples × window_area) / total_datapoints
            window_area = r['window_area']
            total_covered = r['num_samples'] * window_area
            coverage_pct = (total_covered / train_total_datapoints[i]) * 100
            train_coverage.append(coverage_pct)
    
    test_coverage = []
    for i, r in enumerate(test_results):
        if r.get('success'):
            window_area = r['window_area']
            total_covered = r['num_samples'] * window_area
            coverage_pct = (total_covered / test_total_datapoints[i]) * 100
            test_coverage.append(coverage_pct)
    
    print(f"   TRAIN: {np.mean(train_coverage):.1f}% ± {np.std(train_coverage):.1f}%")
    print(f"   TEST:  {np.mean(test_coverage):.1f}% ± {np.std(test_coverage):.1f}%")
    
    # ========================================
    # MOLTIPLICATORE SAMPLES
    # ========================================
    print("\n🔢 MOLTIPLICATORE SAMPLES (Test/Train):")
    multiplier = np.mean(test_samples) / np.mean(train_samples)
    print(f"   Ratio medio: {multiplier:.2f}×")
    print(f"   → Il TEST genera mediamente {multiplier:.1f}× più campioni del TRAIN")
    
    if multiplier > 1:
        extra_samples = np.mean(test_samples) - np.mean(train_samples)
        print(f"   → +{extra_samples:.0f} campioni extra per sessione in media")
    
    # ========================================
    # EFFICIENZA NEURON SLIDING
    # ========================================
    if test_neurons_groups:
        print("\n⚙️  EFFICIENZA NEURON SLIDING (Test):")
        avg_groups = np.mean(test_neurons_groups)
        print(f"   Gruppi neuroni (mean): {avg_groups:.1f}")
        print(f"   Samples per gruppo: {np.mean(test_samples) / avg_groups:.0f}")
        
        # Quanti neuroni totali vengono testati?
        avg_neurons_raw = np.mean([r['total_neurons_raw'] for r in test_results if r.get('success')])
        coverage_neurons_pct = (avg_groups * NEURON_STRIDE / avg_neurons_raw) * 100
        print(f"   Coverage neuroni: {coverage_neurons_pct:.0f}%")
    
    print("="*70)


def plot_train_vs_test_comparison(train_results, test_results, save_dir):
    """
    Crea grafici comparativi Train vs Test.
    """
    import matplotlib.pyplot as plt
    
    save_dir = Path(save_dir)
    
    # ========================================
    # FIGURA: Confronto metriche
    # ========================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('📊 TRAIN vs TEST: Confronto Completo', fontsize=16, fontweight='bold')
    
    # Estrai dati
    train_samples = [r['num_samples'] for r in train_results if r.get('success')]
    test_samples = [r['num_samples'] for r in test_results if r.get('success')]
    
    train_ratios = [r['ratio'] for r in train_results if r.get('success')]
    test_ratios = [r['ratio'] for r in test_results if r.get('success')]
    
    train_timepoints = [r['total_timepoints_raw'] for r in train_results if r.get('success')]
    test_timepoints = [r['total_timepoints_raw'] for r in test_results if r.get('success')]
    
    train_neurons = [r['total_neurons_raw'] for r in train_results if r.get('success')]
    test_neurons = [r['total_neurons_raw'] for r in test_results if r.get('success')]
    
    test_groups = [r['num_neuron_groups'] for r in test_results if r.get('success')]
    
    # 1. Numero di samples
    axes[0, 0].bar(['Train', 'Test'], 
                   [np.mean(train_samples), np.mean(test_samples)],
                   color=['skyblue', 'lightcoral'], alpha=0.7, yerr=[np.std(train_samples), np.std(test_samples)], capsize=5)
    axes[0, 0].set_ylabel('Samples (mean)')
    axes[0, 0].set_title('Samples per Session')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Ratio (samples/area)
    axes[0, 1].bar(['Train', 'Test'], 
                   [np.mean(train_ratios), np.mean(test_ratios)],
                   color=['skyblue', 'lightcoral'], alpha=0.7, yerr=[np.std(train_ratios), np.std(test_ratios)], capsize=5)
    axes[0, 1].set_ylabel('Ratio')
    axes[0, 1].set_title('Ratio (samples/window_area)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Timepoints
    axes[0, 2].bar(['Train', 'Test'], 
                   [np.mean(train_timepoints), np.mean(test_timepoints)],
                   color=['skyblue', 'lightcoral'], alpha=0.7, yerr=[np.std(train_timepoints), np.std(test_timepoints)], capsize=5)
    axes[0, 2].set_ylabel('Timepoints (mean)')
    axes[0, 2].set_title('Timepoints per Session')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # 4. Distribuzione samples
    axes[1, 0].boxplot([train_samples, test_samples], labels=['Train', 'Test'], patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 0].set_ylabel('Samples')
    axes[1, 0].set_title('Samples Distribution')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Neuroni totali
    axes[1, 1].bar(['Train', 'Test'], 
                   [np.mean(train_neurons), np.mean(test_neurons)],
                   color=['skyblue', 'lightcoral'], alpha=0.7, yerr=[np.std(train_neurons), np.std(test_neurons)], capsize=5)
    axes[1, 1].set_ylabel('Total Neurons (mean)')
    axes[1, 1].set_title('Total Neurons Available')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. Gruppi neuroni (solo test)
    if test_groups:
        axes[1, 2].hist(test_groups, bins=10, color='lightcoral', alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(np.mean(test_groups), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(test_groups):.1f}')
        axes[1, 2].set_xlabel('Neuron Groups')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Neuron Groups (Test Sessions)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'train_vs_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Grafico salvato: {save_dir / 'train_vs_test_comparison.png'}")


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
    
    # ========================================
    # 🆕 NUOVE ANALISI
    # ========================================
    
    # 1. Confronto lunghezza sessioni
    compare_session_lengths(train_results, test_results)
    
    # 2. Confronto coverage TRAIN vs TEST
    compare_train_vs_test_coverage(train_results, test_results)
    
    # 3. Tabella comparativa (già esistente)
    print_comparison_table(train_results, test_results)
    
    # 4. 🆕 Grafico comparativo
    save_dir = Path("verification_results")
    save_dir.mkdir(exist_ok=True)
    plot_train_vs_test_comparison(train_results, test_results, save_dir)
    
    # Salva risultati
    import json
    
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
        'test_sessions': test_results,
        'comparison': {
            'train_samples_mean': float(np.mean([r['num_samples'] for r in train_results if r.get('success')])),
            'test_samples_mean': float(np.mean([r['num_samples'] for r in test_results if r.get('success')])),
            'multiplier': float(np.mean([r['num_samples'] for r in test_results if r.get('success')]) / 
                              np.mean([r['num_samples'] for r in train_results if r.get('success')])),
        }
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