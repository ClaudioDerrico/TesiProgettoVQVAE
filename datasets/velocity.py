"""
Single-Session Velocity Dataset - IMPROVED VERSION

Key improvements:
1. Uses instantaneous values instead of window averages
2. Provides multiple target computation options
3. Reduces stride for more training samples
4. Better outlier handling in normalization
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.behavioral import load_behavioral_data_from_session
from datasets.calcium import extract_neural_data_from_hdf5


class SingleSessionVelocityDataset(Dataset):
    """
    Dataset per velocity prediction da UNA singola sessione - VERSIONE MIGLIORATA
    
    Key features:
    - Carica TUTTI i neuroni della sessione
    - Ogni sample = (neuroni, 60) con velocity associata
    - Target più diversificati (non più solo medie)
    
    Args:
        session_id: ID della sessione
        window_size: Dimensione finestra temporale (default: 60)
        stride: Stride per sliding window (default: 15 per più samples)
        normalize: Normalizza velocity (Z-score)
        target_type: Tipo di target ('instant', 'max', 'diff', 'weighted', 'future')
        future_offset: Offset temporale per predizione futura (default: 5)
    """
    
    def __init__(self, session_id, window_size=60, stride=15, normalize=True,
                 target_type='instant', future_offset=5):
        self.session_id = session_id
        self.window_size = window_size
        self.stride = stride  # Ridotto da 30 a 15 per più samples
        self.normalize = normalize
        self.target_type = target_type
        self.future_offset = future_offset
        
        # Storage
        self.neural_windows = []  # Lista di (num_neurons, 60)
        self.velocity_windows = []  # Lista di scalari
        
        # Carica sessione
        self._load_session()
        
        # Normalizza velocity
        if self.normalize and len(self.velocity_windows) > 0:
            self._normalize_velocity()
        
        print(f"\n✅ Session {session_id} dataset created:")
        print(f"   Total windows: {len(self.neural_windows)}")
        print(f"   Neurons per window: {self.neural_windows[0].shape[0]}")
        print(f"   Time per window: {self.neural_windows[0].shape[1]}")
        print(f"   Target type: {target_type}")
        print(f"   Unique velocity values: {len(np.unique(self.velocity_windows))}")
    
    def _load_session(self):
        """Carica dati da una singola sessione"""
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
        
        boc = BrainObservatoryCache()
        
        print(f"\n📊 Loading session {self.session_id}...")
        
        try:
            # ============================================================
            # NEURAL DATA - TUTTI I NEURONI
            # ============================================================
            neural_result = extract_neural_data_from_hdf5(self.session_id)
            
            if neural_result is None or not neural_result['success']:
                raise RuntimeError(f"Failed to load neural data for session {self.session_id}")
            
            dff_data = neural_result['dff_data']
            
            # Assicura (neurons, time)
            if dff_data.shape[0] > dff_data.shape[1]:
                dff_data = dff_data.T
            
            num_neurons = dff_data.shape[0]
            total_time = dff_data.shape[1]
            
            print(f"   Neural: {dff_data.shape} ({num_neurons} neurons)")
            
            # ============================================================
            # BEHAVIORAL DATA - VELOCITY
            # ============================================================
            behavioral_data = load_behavioral_data_from_session(self.session_id, boc)
            
            if behavioral_data is None:
                raise RuntimeError(f"Failed to load behavioral data for session {self.session_id}")
            
            # Verifica allineamento
            behavioral_time = len(behavioral_data['timestamps'])
            
            if total_time != behavioral_time:
                print(f"   ⚠️ Mismatch: neural={total_time}, behavioral={behavioral_time}")
                # Tronca al minimo
                min_time = min(total_time, behavioral_time)
                dff_data = dff_data[:, :min_time]
                behavioral_data['running_speed'] = behavioral_data['running_speed'][:min_time]
            
            # ============================================================
            # CREATE WINDOWS - TUTTI I NEURONI PER FINESTRA
            # ============================================================
            self._create_windows(dff_data, behavioral_data['running_speed'])
            
            print(f"   ✅ Created {len(self.neural_windows)} windows")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_windows(self, neural_data, velocity_data):
        """
        Crea sliding windows con TUTTI i neuroni - TARGET DIVERSIFICATI
        
        Args:
            neural_data: (neurons, time)
            velocity_data: (time,)
        """
        total_time = neural_data.shape[1]
        
        # Calcola quante finestre possiamo creare
        max_end = total_time - self.future_offset if self.target_type == 'future' else total_time
        
        # Sliding window
        for start in range(0, min(max_end, total_time) - self.window_size + 1, self.stride):
            end = start + self.window_size
            
            # Neural window: TUTTI i neuroni
            neural_window = neural_data[:, start:end]  # (neurons, 60)
            
            # ========================================================================
            # 🔥 TARGET DIVERSIFICATI - Scelta strategia
            # ========================================================================
            
            if self.target_type == 'instant':
                # Usa l'ULTIMO valore della finestra (predizione istantanea)
                velocity_value = velocity_data[end - 1]
                
            elif self.target_type == 'future':
                # Predici il futuro (oltre la finestra)
                future_idx = min(end + self.future_offset - 1, len(velocity_data) - 1)
                velocity_value = velocity_data[future_idx]
                
            elif self.target_type == 'max':
                # Usa il MASSIMO ASSOLUTO nella finestra (cattura picchi di attività)
                velocity_value = np.max(np.abs(velocity_data[start:end])) * np.sign(velocity_data[end-1])
                
            elif self.target_type == 'diff':
                # Usa la DIFFERENZA (accelerazione/decelerazione)
                velocity_value = velocity_data[end - 1] - velocity_data[start]
                
            elif self.target_type == 'weighted':
                # Media pesata esponenziale (più peso ai valori recenti)
                weights = np.exp(np.linspace(-2, 0, self.window_size))
                weights = weights / weights.sum()
                velocity_value = np.average(velocity_data[start:end], weights=weights)
                
            elif self.target_type == 'median':
                # Mediana (robusta agli outliers)
                velocity_value = np.median(velocity_data[start:end])
                
            elif self.target_type == 'std':
                # Deviazione standard (cattura variabilità)
                velocity_value = np.std(velocity_data[start:end])
                
            else:  # 'mean' o default
                # Media semplice (comportamento originale)
                velocity_value = np.mean(velocity_data[start:end])
            
            # Aggiungi solo se non è NaN
            if not np.isnan(velocity_value):
                self.neural_windows.append(neural_window)
                self.velocity_windows.append(velocity_value)
    
    def _normalize_velocity(self):
        """Normalizzazione robusta usando IQR per rimuovere outliers"""
        velocity_array = np.array(self.velocity_windows)
        
        print(f"\n   🔍 Original velocity statistics:")
        print(f"      Count: {len(velocity_array)}")
        print(f"      Min: {velocity_array.min():.4f}, Max: {velocity_array.max():.4f}")
        print(f"      Mean: {velocity_array.mean():.4f}, Std: {velocity_array.std():.4f}")
        print(f"      Median: {np.median(velocity_array):.4f}")
        print(f"      Unique values: {len(np.unique(velocity_array))}")
        
        # ========================================================================
        # ROBUST CLIPPING usando IQR (Interquartile Range)
        # ========================================================================
        q1 = np.percentile(velocity_array, 25)
        q3 = np.percentile(velocity_array, 75)
        iqr = q3 - q1
        
        # Definisci bounds usando IQR (più robusto dei percentili fissi)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        print(f"\n   📊 IQR-based clipping:")
        print(f"      Q1 (25th percentile): {q1:.4f}")
        print(f"      Q3 (75th percentile): {q3:.4f}")
        print(f"      IQR: {iqr:.4f}")
        print(f"      Lower bound: {lower_bound:.4f}")
        print(f"      Upper bound: {upper_bound:.4f}")
        
        # Clip outliers
        velocity_array_clipped = np.clip(velocity_array, lower_bound, upper_bound)
        
        n_clipped = np.sum(velocity_array != velocity_array_clipped)
        pct_clipped = 100 * n_clipped / len(velocity_array)
        
        print(f"\n   ✅ After clipping:")
        print(f"      Clipped {n_clipped} values ({pct_clipped:.1f}%)")
        print(f"      New range: [{velocity_array_clipped.min():.4f}, {velocity_array_clipped.max():.4f}]")
        
        # ========================================================================
        # Z-SCORE NORMALIZATION sui dati clippati
        # ========================================================================
        self.velocity_mean = np.mean(velocity_array_clipped)
        self.velocity_std = np.std(velocity_array_clipped)
        
        print(f"\n   📊 Normalization parameters:")
        print(f"      Mean (after clip): {self.velocity_mean:.4f}")
        print(f"      Std (after clip): {self.velocity_std:.4f}")
        
        if self.velocity_std < 1e-8:
            print(f"      ⚠️ WARNING: Std too small, setting to 1.0")
            self.velocity_std = 1.0
        
        # Normalize
        velocity_array_norm = (velocity_array_clipped - self.velocity_mean) / self.velocity_std
        
        self.velocity_windows = velocity_array_norm.tolist()
        
        # ========================================================================
        # FINAL CHECK
        # ========================================================================
        print(f"\n   📊 After normalization:")
        print(f"      Range: [{min(self.velocity_windows):.4f}, {max(self.velocity_windows):.4f}]")
        print(f"      Mean: {np.mean(self.velocity_windows):.6f}")
        print(f"      Std: {np.std(self.velocity_windows):.6f}")
        print(f"      Unique normalized values: {len(np.unique(self.velocity_windows))}")
        
        # Check if range is reasonable
        max_abs = max(abs(min(self.velocity_windows)), abs(max(self.velocity_windows)))
        
        if max_abs > 5.0:
            print(f"      ⚠️ WARNING: Large range ±{max_abs:.2f}, consider checking data")
        else:
            print(f"      ✅ Range is reasonable (±{max_abs:.2f} std)")
        
        print(f"\n   ✅ Normalization complete: {len(self.velocity_windows)} windows ready")
    
    def __len__(self):
        return len(self.neural_windows)
    
    def __getitem__(self, idx):
        """
        Returns:
            neural_data: (num_neurons, window_size)
            velocity: scalar
        """
        neural = torch.FloatTensor(self.neural_windows[idx])
        velocity = torch.FloatTensor([self.velocity_windows[idx]]).squeeze()
        
        return neural, velocity


def create_single_session_dataloaders(session_id, test_split=0.2, 
                                     window_size=60, stride=15,  # Stride ridotto!
                                     num_workers=0, normalize=True,
                                     target_type='instant',  # Nuovo parametro
                                     future_offset=5):
    """
    Factory function per creare dataloaders da UNA sessione - VERSIONE MIGLIORATA
    
    Key: Batch size = numero di neuroni!
    
    Args:
        session_id: Session ID
        test_split: Frazione per test
        window_size: Window size
        stride: Stride (default ridotto a 15 per più samples)
        num_workers: Workers
        normalize: Normalizza velocity
        target_type: Tipo di target ('instant', 'future', 'max', 'diff', 'weighted', 'median', 'std', 'mean')
        future_offset: Offset per predizione futura (se target_type='future')
    
    Returns:
        train_loader, test_loader, dataset_info
    """
    
    print("="*70)
    print(f"🎯 CREATING IMPROVED VELOCITY DATALOADERS")
    print(f"   Session ID: {session_id}")
    print(f"   Target type: {target_type}")
    print(f"   Stride: {stride} (more samples)")
    print("="*70)
    
    # Create dataset
    dataset = SingleSessionVelocityDataset(
        session_id=session_id,
        window_size=window_size,
        stride=stride,
        normalize=normalize,
        target_type=target_type,
        future_offset=future_offset
    )
    
    # Get num_neurons (batch size)
    num_neurons = dataset.neural_windows[0].shape[0]
    
    # Split
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Dataloaders - BATCH SIZE = 1 window at a time
    # Perché ogni window ha già tutti i neuroni
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,  # ⚠️ 1 window = tutti i neuroni
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )
    
    dataset_info = {
        'session_id': session_id,
        'total_samples': total_size,
        'train_samples': train_size,
        'test_samples': test_size,
        'num_neurons': num_neurons,
        'window_size': window_size,
        'stride': stride,
        'normalized': normalize,
        'target_type': target_type,
        'unique_targets': len(np.unique(dataset.velocity_windows))
    }
    
    print(f"\n✅ Dataloaders created!")
    print(f"   Train: {train_size} windows")
    print(f"   Test: {test_size} windows")
    print(f"   Neurons per window: {num_neurons}")
    print(f"   Unique velocity targets: {dataset_info['unique_targets']}")
    print("="*70)
    
    return train_loader, test_loader, dataset_info


# ============================================================================
# TEST DIVERSI TARGET TYPES
# ============================================================================

if __name__ == "__main__":
    from datasets.calcium import TRAINING_SESSION_IDS
    
    session_id = TRAINING_SESSION_IDS[0]
    
    print("🧪 Testing different target types for diversity\n")
    
    target_types = ['instant', 'future', 'max', 'diff', 'weighted', 'mean']
    
    results = {}
    
    for target_type in target_types:
        print(f"\n📊 Testing target_type = '{target_type}'")
        print("-"*50)
        
        train_loader, test_loader, info = create_single_session_dataloaders(
            session_id=session_id,
            target_type=target_type,
            stride=10,  # Piccolo stride per più samples
            test_split=0.2
        )
        
        results[target_type] = info['unique_targets']
        
        # Controlla il primo batch
        neural, velocity = next(iter(train_loader))
        print(f"   First batch velocity: {velocity.item():.4f}")
    
    print("\n" + "="*70)
    print("📊 SUMMARY - Unique targets by type:")
    print("="*70)
    for target_type, unique_count in results.items():
        print(f"   {target_type:10s}: {unique_count:4d} unique values")
    
    best_type = max(results, key=results.get)
    print(f"\n✅ Best target type: '{best_type}' with {results[best_type]} unique values")
    print("="*70)