#!/usr/bin/env python3
"""
Quick check: quale sessioni hanno running speed data

Usage:
    python scripts/quick_check_running_speed.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np

# Le tue 10 sessioni di training
TRAINING_SESSIONS = [
    650389887,  # VISp
    642883713,  # VISp
    501704220,  # VISp
    501271265,  # VISp
    502205092,  # VISp
    501773889,  # VISp
    502066273,  # VISp
    502115959,  # VISp
    503109347,  # VISp
    502608215,  # VISp
]

print("="*70)
print("🔍 CHECKING RUNNING SPEED DATA FOR ALL TRAINING SESSIONS")
print("="*70)

boc = BrainObservatoryCache(manifest_file='brain_observatory/manifest.json')

results = []

for i, session_id in enumerate(TRAINING_SESSIONS, 1):
    print(f"\n[{i}/10] Session {session_id}...", end=" ")
    
    try:
        dataset = boc.get_ophys_experiment_data(session_id)
        
        # Check metadata
        metadata = dataset.get_metadata()
        session_type = metadata.get('session_type', 'unknown')
        targeted_structure = metadata.get('targeted_structure', 'unknown')
        
        # Initialize result dict
        session_result = {
            'session_id': session_id,
            'session_type': session_type,
            'structure': targeted_structure,
            'has_running': False,
            'has_pupil_size': False,
            'has_pupil_location': False,
            'behavioral_count': 0
        }
        
        print()  # Newline for clarity
        
        # ====================================================================
        # 1. RUNNING SPEED
        # ====================================================================
        try:
            running_speed, timestamps = dataset.get_running_speed()
            
            valid_samples = np.sum(~np.isnan(running_speed))
            total_samples = len(running_speed)
            valid_pct = 100 * valid_samples / total_samples
            
            speed_clean = running_speed[~np.isnan(running_speed)]
            
            if len(speed_clean) > 0 and np.std(speed_clean) > 0.1:
                speed_mean = np.mean(speed_clean)
                speed_std = np.std(speed_clean)
                speed_max = np.max(speed_clean)
                
                print(f"      ✅ Running Speed: {valid_pct:.1f}% valid, "
                      f"mean={speed_mean:.2f}, std={speed_std:.2f}")
                
                session_result['has_running'] = True
                session_result['running_valid_pct'] = valid_pct
                session_result['running_mean'] = speed_mean
                session_result['running_std'] = speed_std
                session_result['running_max'] = speed_max
                session_result['behavioral_count'] += 1
            else:
                print(f"      ❌ Running Speed: no variance")
                
        except Exception as e:
            print(f"      ❌ Running Speed: not available")
        
        # ====================================================================
        # 2. PUPIL SIZE (area/diameter)
        # ====================================================================
        try:
            pupil_size = dataset.get_pupil_size()
            
            if pupil_size is not None:
                pupil_array = np.array(pupil_size)
                valid_pupil = ~np.isnan(pupil_array)
                valid_samples = np.sum(valid_pupil)
                total_samples = len(pupil_array)
                valid_pct = 100 * valid_samples / total_samples
                
                if valid_samples > 100:
                    pupil_clean = pupil_array[valid_pupil]
                    
                    if np.std(pupil_clean) > 1.0:  # At least some variance
                        pupil_mean = np.mean(pupil_clean)
                        pupil_std = np.std(pupil_clean)
                        
                        print(f"      ✅ Pupil Size: {valid_pct:.1f}% valid, "
                              f"mean={pupil_mean:.2f}, std={pupil_std:.2f}")
                        
                        session_result['has_pupil_size'] = True
                        session_result['pupil_valid_pct'] = valid_pct
                        session_result['pupil_mean'] = pupil_mean
                        session_result['pupil_std'] = pupil_std
                        session_result['behavioral_count'] += 1
                    else:
                        print(f"      ❌ Pupil Size: no variance")
                else:
                    print(f"      ❌ Pupil Size: insufficient valid samples")
            else:
                print(f"      ❌ Pupil Size: not available")
                
        except Exception as e:
            print(f"      ❌ Pupil Size: {str(e)[:50]}")
        
        # ====================================================================
        # 3. PUPIL LOCATION (eye tracking - X and Y position)
        # ====================================================================
        try:
            pupil_location = dataset.get_pupil_location()
            
            if pupil_location is not None and len(pupil_location) > 0:
                # Check if it's a DataFrame or array
                if hasattr(pupil_location, 'shape'):
                    # It's an array-like
                    if len(pupil_location.shape) >= 2 and pupil_location.shape[1] >= 2:
                        eye_x = pupil_location[:, 0]
                        eye_y = pupil_location[:, 1]
                        
                        # Check validity
                        valid_x = ~np.isnan(eye_x)
                        valid_y = ~np.isnan(eye_y)
                        valid_both = valid_x & valid_y
                        
                        valid_samples = np.sum(valid_both)
                        total_samples = len(eye_x)
                        valid_pct = 100 * valid_samples / total_samples
                        
                        if valid_samples > 100:
                            eye_x_clean = eye_x[valid_both]
                            eye_y_clean = eye_y[valid_both]
                            
                            x_std = np.std(eye_x_clean)
                            y_std = np.std(eye_y_clean)
                            
                            if x_std > 1.0 or y_std > 1.0:  # At least some movement
                                x_mean = np.mean(eye_x_clean)
                                y_mean = np.mean(eye_y_clean)
                                
                                print(f"      ✅ Eye Position: {valid_pct:.1f}% valid")
                                print(f"         X: mean={x_mean:.2f}, std={x_std:.2f}")
                                print(f"         Y: mean={y_mean:.2f}, std={y_std:.2f}")
                                
                                session_result['has_pupil_location'] = True
                                session_result['eye_valid_pct'] = valid_pct
                                session_result['eye_x_mean'] = x_mean
                                session_result['eye_x_std'] = x_std
                                session_result['eye_y_mean'] = y_mean
                                session_result['eye_y_std'] = y_std
                                session_result['behavioral_count'] += 2  # X and Y count as 2
                            else:
                                print(f"      ❌ Eye Position: no movement variance")
                        else:
                            print(f"      ❌ Eye Position: insufficient valid samples")
                    else:
                        print(f"      ❌ Eye Position: unexpected shape {pupil_location.shape}")
                else:
                    print(f"      ❌ Eye Position: unexpected format")
            else:
                print(f"      ❌ Eye Position: not available")
                
        except Exception as e:
            print(f"      ❌ Eye Position: {str(e)[:50]}")
        
        results.append(session_result)
            
    except Exception as e:
        print(f"❌ Error loading session: {e}")
        results.append({
            'session_id': session_id,
            'has_running': False,
            'error': str(e)
        })

# Summary
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

sessions_with_running = [r for r in results if r.get('has_running', False)]
sessions_with_pupil_size = [r for r in results if r.get('has_pupil_size', False)]
sessions_with_pupil_location = [r for r in results if r.get('has_pupil_location', False)]

# Sessions with ANY behavioral data
sessions_with_any = [r for r in results if r.get('behavioral_count', 0) > 0]
sessions_with_all_4 = [r for r in results if r.get('behavioral_count', 0) >= 4]

print(f"\n📈 Behavioral Data Availability:")
print(f"   Running Speed:    {len(sessions_with_running)}/10 sessions")
print(f"   Pupil Size:       {len(sessions_with_pupil_size)}/10 sessions")
print(f"   Eye Position:     {len(sessions_with_pupil_location)}/10 sessions")
print(f"\n   ANY behavioral:   {len(sessions_with_any)}/10 sessions")
print(f"   ALL 4 variables:  {len(sessions_with_all_4)}/10 sessions")

if sessions_with_any:
    print(f"\n🎯 DETAILED BREAKDOWN:")
    print("-"*70)
    
    for r in results:
        count = r.get('behavioral_count', 0)
        if count > 0:
            components = []
            if r.get('has_running'): components.append("running")
            if r.get('has_pupil_size'): components.append("pupil_size")
            if r.get('has_pupil_location'): components.append("eye_x/y")
            
            print(f"   Session {r['session_id']}: {count}/4 variables → {', '.join(components)}")

if sessions_with_running:
    print(f"\n🎯 USABLE SESSIONS (at least running speed):")
    print("   BEHAVIORAL_TRAINING_SESSIONS = [")
    for r in sessions_with_running:
        components = []
        if r.get('has_running'): components.append("run")
        if r.get('has_pupil_size'): components.append("pupil")
        if r.get('has_pupil_location'): components.append("eye")
        
        print(f"       {r['session_id']},  # {'/'.join(components)} - "
              f"{r.get('structure', '?')} - mean_speed={r.get('running_mean', 0):.1f}")
    print("   ]")
    
    # Statistics
    if len(sessions_with_running) > 0:
        print(f"\n📈 Running Speed Statistics:")
        all_means = [r['running_mean'] for r in sessions_with_running if 'running_mean' in r]
        all_stds = [r['running_std'] for r in sessions_with_running if 'running_std' in r]
        all_maxs = [r['running_max'] for r in sessions_with_running if 'running_max' in r]
        
        if all_means:
            print(f"   Average mean speed: {np.mean(all_means):.2f} cm/s")
            print(f"   Average std: {np.mean(all_stds):.2f} cm/s")
            print(f"   Max speed observed: {np.max(all_maxs):.2f} cm/s")

else:
    print(f"\n⚠️ NO USABLE SESSIONS")
    for r in results:
        print(f"   {r['session_id']} - {r.get('structure', '?')} - No behavioral data")

# Save results
print(f"\n💾 Saving results to running_speed_check.txt...")
with open('running_speed_check.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("RUNNING SPEED DATA CHECK - TRAINING SESSIONS\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Sessions with running speed: {len(sessions_with_running)}/10\n\n")
    
    for r in results:
        f.write(f"Session {r['session_id']}:\n")
        f.write(f"  Structure: {r.get('structure', 'unknown')}\n")
        f.write(f"  Session type: {r.get('session_type', 'unknown')}\n")
        f.write(f"  Has running speed: {'YES' if r.get('has_running') else 'NO'}\n")
        if 'mean' in r:
            f.write(f"  Mean speed: {r['mean']:.2f} cm/s\n")
            f.write(f"  Std: {r['std']:.2f} cm/s\n")
            f.write(f"  Max: {r['max']:.2f} cm/s\n")
        f.write("\n")

print("✅ Results saved!")

print("\n" + "="*70)
print("💡 RECOMMENDATION:")
print("="*70)

if len(sessions_with_all_4) > 0:
    print(f"🌟 EXCELLENT! You have {len(sessions_with_all_4)} sessions with ALL 4 behavioral variables!")
    print(f"   These are PERFECT for full behavioral prediction training!")
    print(f"   Prediction targets: pupil size, eye X, eye Y, running speed")
    
elif len(sessions_with_any) >= 5:
    print(f"✅ GOOD! You have {len(sessions_with_any)} sessions with SOME behavioral data!")
    
    # Count what we have
    total_running = len(sessions_with_running)
    total_pupil = len(sessions_with_pupil_size)
    total_eye = len(sessions_with_pupil_location)
    
    print(f"\n   Available data:")
    print(f"   - Running speed: {total_running} sessions")
    print(f"   - Pupil size: {total_pupil} sessions")
    print(f"   - Eye position: {total_eye} sessions")
    
    print(f"\n   STRATEGY:")
    if total_running >= 5:
        print(f"   ✓ Focus on predicting RUNNING SPEED (most available)")
    if total_pupil >= 3:
        print(f"   ✓ Can also try predicting PUPIL SIZE")
    if total_eye >= 3:
        print(f"   ✓ Can also try predicting EYE POSITION")
    
    print(f"\n   Variables with insufficient data will be set to zeros")
    print(f"   Model will learn meaningful predictions for available variables")
    
elif len(sessions_with_any) > 0:
    print(f"⚠️ LIMITED: You have {len(sessions_with_any)} sessions with behavioral data")
    print(f"   This might work but results may be limited")
    print(f"   Consider:")
    print(f"   1. Finding more sessions with behavioral data")
    print(f"   2. Focus only on the most available variable (likely running speed)")
    
else:
    print(f"❌ NO sessions with ANY behavioral data found!")
    print(f"   Options:")
    print(f"   1. Try different session IDs")
    print(f"   2. Use Visual Behavior dataset instead")
    print(f"   3. Check if data needs to be downloaded separately")

print("="*70)