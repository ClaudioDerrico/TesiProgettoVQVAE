#!/usr/bin/env python3
"""
Check quale sessioni Allen Brain hanno dati comportamentali disponibili

Usage:
    python scripts/check_behavioral_data_availability.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from datasets.calcium import TRAINING_SESSION_IDS, TEST_SESSION_IDS
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

def check_session_behavioral_data(session_id, boc):
    """
    Verifica se una sessione ha dati comportamentali
    
    Returns:
        dict con info disponibilità
    """
    try:
        dataset = boc.get_ophys_experiment_data(session_id)
        nwb_file = dataset.nwb_file
        
        info = {
            'session_id': session_id,
            'has_pupil': False,
            'has_running': False,
            'pupil_samples': 0,
            'running_samples': 0,
        }
        
        with h5py.File(nwb_file, 'r') as f:
            # Check pupil
            pupil_path = 'processing/brain_observatory_pipeline/EyeTracking'
            if pupil_path in f:
                pupil_area_path = f'{pupil_path}/pupil_area/data'
                if pupil_area_path in f:
                    pupil_data = f[pupil_area_path][:]
                    valid_pupil = ~np.isnan(pupil_data)
                    if np.sum(valid_pupil) > 100:  # At least 100 valid samples
                        info['has_pupil'] = True
                        info['pupil_samples'] = np.sum(valid_pupil)
            
            # Check running speed
            running_path = 'processing/brain_observatory_pipeline/RunningSpeed'
            if running_path in f:
                running_data_path = f'{running_path}/data'
                if running_data_path in f:
                    running_data = f[running_data_path][:]
                    valid_running = ~np.isnan(running_data)
                    if np.sum(valid_running) > 100:
                        info['has_running'] = True
                        info['running_samples'] = np.sum(valid_running)
        
        return info
        
    except Exception as e:
        print(f"   ❌ Error checking session {session_id}: {e}")
        return None


def main():
    print("="*70)
    print("🔍 CHECKING BEHAVIORAL DATA AVAILABILITY")
    print("="*70)
    
    boc = BrainObservatoryCache()
    
    all_sessions = TRAINING_SESSION_IDS + TEST_SESSION_IDS
    
    print(f"\n📊 Checking {len(all_sessions)} sessions...")
    print(f"   Training: {len(TRAINING_SESSION_IDS)}")
    print(f"   Test: {len(TEST_SESSION_IDS)}")
    
    # Check all sessions
    results = []
    
    for i, session_id in enumerate(all_sessions, 1):
        print(f"\n[{i}/{len(all_sessions)}] Session {session_id}...", end=" ")
        
        info = check_session_behavioral_data(session_id, boc)
        
        if info:
            results.append(info)
            
            has_data = info['has_pupil'] or info['has_running']
            
            if has_data:
                print("✅", end=" ")
                if info['has_pupil']:
                    print(f"Pupil({info['pupil_samples']})", end=" ")
                if info['has_running']:
                    print(f"Running({info['running_samples']})", end=" ")
                print()
            else:
                print("❌ No behavioral data")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    sessions_with_pupil = [r for r in results if r['has_pupil']]
    sessions_with_running = [r for r in results if r['has_running']]
    sessions_with_any = [r for r in results if r['has_pupil'] or r['has_running']]
    sessions_with_both = [r for r in results if r['has_pupil'] and r['has_running']]
    
    print(f"\n✅ Sessions with ANY behavioral data: {len(sessions_with_any)}/{len(results)}")
    print(f"   - With pupil data: {len(sessions_with_pupil)}")
    print(f"   - With running speed: {len(sessions_with_running)}")
    print(f"   - With BOTH: {len(sessions_with_both)}")
    
    # Print sessions with data
    if sessions_with_any:
        print(f"\n🎯 Sessions to use for behavioral training:")
        print("   Session IDs with behavioral data:")
        
        # TRAINING sessions
        train_with_data = [r for r in sessions_with_any if r['session_id'] in TRAINING_SESSION_IDS]
        if train_with_data:
            print(f"\n   📚 TRAINING ({len(train_with_data)} sessions):")
            print("   BEHAVIORAL_TRAINING_SESSIONS = [")
            for r in train_with_data:
                pupil_str = "✓pupil" if r['has_pupil'] else ""
                running_str = "✓running" if r['has_running'] else ""
                print(f"       {r['session_id']},  # {pupil_str} {running_str}")
            print("   ]")
        
        # TEST sessions
        test_with_data = [r for r in sessions_with_any if r['session_id'] in TEST_SESSION_IDS]
        if test_with_data:
            print(f"\n   🔬 TEST ({len(test_with_data)} sessions):")
            print("   BEHAVIORAL_TEST_SESSIONS = [")
            for r in test_with_data:
                pupil_str = "✓pupil" if r['has_pupil'] else ""
                running_str = "✓running" if r['has_running'] else ""
                print(f"       {r['session_id']},  # {pupil_str} {running_str}")
            print("   ]")
    else:
        print("\n❌ No sessions with behavioral data found!")
        print("   This might mean:")
        print("   - Data not downloaded yet")
        print("   - Different Allen Brain dataset version")
        print("   - Need to use different session IDs")
    
    # Save results
    print(f"\n💾 Saving results to behavioral_data_availability.txt...")
    
    with open('behavioral_data_availability.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("BEHAVIORAL DATA AVAILABILITY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total sessions checked: {len(results)}\n")
        f.write(f"Sessions with behavioral data: {len(sessions_with_any)}\n\n")
        
        f.write("SESSIONS WITH BEHAVIORAL DATA:\n")
        f.write("-"*70 + "\n")
        
        for r in sessions_with_any:
            session_type = "TRAIN" if r['session_id'] in TRAINING_SESSION_IDS else "TEST"
            f.write(f"{r['session_id']} ({session_type})\n")
            if r['has_pupil']:
                f.write(f"  ✓ Pupil: {r['pupil_samples']} samples\n")
            if r['has_running']:
                f.write(f"  ✓ Running: {r['running_samples']} samples\n")
            f.write("\n")
    
    print("✅ Results saved!")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()