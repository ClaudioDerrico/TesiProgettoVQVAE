"""
Sequence Behavioral Dataset

Restituisce sequenze temporali COMPLETE di velocità (60 timesteps)
invece di un singolo valore medio.

Questo permette al modello di imparare i pattern temporali e i picchi istantanei!
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.behavioral import load_behavioral_data_from_session
from datasets.calcium import extract_neural_data_from_hdf5


class SequenceBehavioralDataset(Dataset):
    """
    Dataset per predizione di sequenze temporali comportamentali
    
    Key difference da AllenBehavioralDataset:
    - Output: (60,) sequenza temporale completa di velocità
    - Non usa MEAN, mantiene tutti i 60 valori!
    
    Args:
        session_ids: Lista session IDs
        window_size: Dimensione finestra (default: 60)
        stride: Stride per sliding window
        normalize: Normalizza velocity
    """
    
    def __init__(self, session_ids, window_size=60, stride=30, normalize=True):
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        
        # Storage
        self.neural_windows = []  # (num_neurons, 60)
        self.velocity_sequences = []  # (60,) - SEQUENZA COMPLETA!
        
        # Carica sessioni
        self._load_all_sessions(session_ids)
        
        # Normalizza
        if self.normalize and len(self.velocity_sequences) > 0:
            self._normalize_sequences()
        
        print(f"\n✅ Sequence Dataset creato:")
        print(f"   Total windows: {len(self.neural_windows)}")
        print(f"   Neural shape: ({self.neural_windows[0].shape[0]}, {self.window_size})")
        print(f"   Velocity shape: ({self.window_size},) - FULL SEQUENCE!")
    
    def _load_all_sessions(self, session_ids):
        """Carica tutte le sessioni"""
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
        
        boc = BrainObservatoryCache()
        
        print(f"\n📊 Loading {len(session_ids)} sessions for sequence prediction...")
        
        for i, session_id in enumerate(session_ids, 1):
            print(f"\n[{i}/{len(session_ids)}] Session {session_id}")
            
            try:
                # ============================================================
                # NEURAL DATA - TUTTI I NEURONI
                # ============================================================
                neural_result = extract_neural_data_from_hdf5(session_id)
                
                if neural_result is None or not neural_result['success']:
                    print(f"   ❌ Failed to load neural data")
                    continue
                
                dff_data = neural_result['dff_data']
                
                # Assicura (neurons, time)
                if dff_data.shape[0] > dff_data.shape[1]:
                    dff_data = dff_data.T
                
                num_neurons = dff_data.shape[0]
                total_time = dff_data.shape[1]
                
                print(f"   Neural: {dff_data.shape} ({num_neurons} neurons)")
                
                # ============================================================
                # BEHAVIORAL DATA - VELOCITY SEQUENCE
                # ============================================================
                behavioral_data = load_behavioral_data_from_session(session_id, boc)
                
                if behavioral_data is None:
                    print(f"   ❌ Failed to load behavioral data")
                    continue
                
                # Verifica allineamento
                behavioral_time = len(behavioral_data['timestamps'])
                
                if total_time != behavioral_time:
                    print(f"   ⚠️ Mismatch: neural={total_time}, behavioral={behavioral_time}")
                    min_time = min(total_time, behavioral_time)
                    dff_data = dff_data[:, :min_time]
                    behavioral_data['running_speed'] = behavioral_data['running_speed'][:min_time]
                
                # ============================================================
                # CREATE WINDOWS - MANTIENI SEQUENZA TEMPORALE COMPLETA
                # ============================================================
                self._create_sequence_windows(dff_data, behavioral_data['running_speed'])
                
                print(f"   ✅ Created {len(self.neural_windows)} total windows")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    def _create_sequence_windows(self, neural_data, velocity_data):
        """
        Crea sliding windows mantenendo SEQUENZA TEMPORALE COMPLETA
        
        Args:
            neural_data: (neurons, time)
            velocity_data: (time,) - velocità per ogni timestep
        """
        total_time = neural_data.shape[1]
        
        # Sliding window
        for start in range(0, total_time - self.window_size + 1, self.stride):
            end = start + self.window_size
            
            # Neural window: (neurons, 60)
            neural_window = neural_data[:, start:end]
            
            # ✅ VELOCITY SEQUENCE: (60,) - TUTTI I VALORI!
            velocity_sequence = velocity_data[start:end]
            
            # Verifica che non ci siano troppi NaN
            valid_count = np.sum(~np.isnan(velocity_sequence))
            
            if valid_count > 0.8 * self.window_size:  # Almeno 80% validi
                # Riempi eventuali NaN con interpolazione
                if np.any(np.isnan(velocity_sequence)):
                    # Interpolazione lineare per NaN
                    nans = np.isnan(velocity_sequence)
                    x = np.arange(len(velocity_sequence))
                    velocity_sequence[nans] = np.interp(
                        x[nans], x[~nans], velocity_sequence[~nans]
                    )
                
                self.neural_windows.append(neural_window)
                self.velocity_sequences.append(velocity_sequence)
    
    def _normalize_sequences(self):
        """Normalizza sequenze di velocità (Z-score globale)"""
        # Stack tutte le sequenze
        all_velocities = np.concatenate(self.velocity_sequences)  # (N*60,)
        
        print(f"\n   📊 Sequence normalization:")
        print(f"      Total timesteps: {len(all_velocities)}")
        print(f"      Original range: [{all_velocities.min():.3f}, {all_velocities.max():.3f}]")
        
        # Robust normalization con clipping IQR
        q1 = np.percentile(all_velocities, 25)
        q3 = np.percentile(all_velocities, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        all_velocities_clipped = np.clip(all_velocities, lower_bound, upper_bound)
        
        # Z-score
        self.velocity_mean = np.mean(all_velocities_clipped)
        self.velocity_std = np.std(all_velocities_clipped)
        
        if self.velocity_std < 1e-8:
            self.velocity_std = 1.0
        
        print(f"      Mean: {self.velocity_mean:.3f}, Std: {self.velocity_std:.3f}")
        
        # Normalizza tutte le sequenze
        normalized_sequences = []
        for seq in self.velocity_sequences:
            seq_clipped = np.clip(seq, lower_bound, upper_bound)
            seq_norm = (seq_clipped - self.velocity_mean) / self.velocity_std
            normalized_sequences.append(seq_norm)
        
        self.velocity_sequences = normalized_sequences
        
        # Check finale
        all_norm = np.concatenate(self.velocity_sequences)
        print(f"      After norm: mean={all_norm.mean():.6f}, std={all_norm.std():.6f}")
        print(f"      Range: [{all_norm.min():.3f}, {all_norm.max():.3f}]")
    
    def __len__(self):
        return len(self.neural_windows)
    
    def __getitem__(self, idx):
        """
        Returns:
            neural_data: (num_neurons, 60)
            velocity_sequence: (60,) - SEQUENZA COMPLETA!
        """
        neural = torch.FloatTensor(self.neural_windows[idx])
        velocity_seq = torch.FloatTensor(self.velocity_sequences[idx])
        
        return neural, velocity_seq


def create_sequence_dataloaders(session_ids, batch_size=32, test_split=0.2,
                                window_size=60, stride=30, num_workers=0,
                                normalize=True):
    """
    Factory function per creare dataloaders con sequenze temporali
    
    Returns:
        train_loader, test_loader, dataset_info
    """
    
    print("="*70)
    print("🧠 CREATING SEQUENCE BEHAVIORAL DATALOADERS")
    print("="*70)
    
    # Create dataset
    dataset = SequenceBehavioralDataset(
        session_ids=session_ids,
        window_size=window_size,
        stride=stride,
        normalize=normalize
    )
    
    # Split
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Info
    neural_shape = dataset.neural_windows[0].shape
    velocity_shape = (window_size,)
    
    dataset_info = {
        'total_samples': total_size,
        'train_samples': train_size,
        'test_samples': test_size,
        'num_sessions': len(session_ids),
        'neural_shape': neural_shape,
        'velocity_shape': velocity_shape,
        'window_size': window_size,
        'stride': stride,
        'normalized': normalize
    }
    
    print(f"\n✅ Sequence Dataloaders created!")
    print(f"   Train: {train_size}")
    print(f"   Test: {test_size}")
    print(f"   Neural: {neural_shape}")
    print(f"   Velocity: {velocity_shape} - FULL TEMPORAL SEQUENCE!")
    print("="*70)
    
    return train_loader, test_loader, dataset_info


if __name__ == "__main__":
    print("🧪 Testing SequenceBehavioralDataset\n")
    
    from datasets.calcium import TRAINING_SESSION_IDS
    
    train_loader, test_loader, info = create_sequence_dataloaders(
        session_ids=[TRAINING_SESSION_IDS[0]],
        batch_size=8,
        test_split=0.2,
        window_size=60,
        stride=30
    )
    
    # Test batch
    neural, velocity_seq = next(iter(train_loader))
    
    print(f"\n📊 Sample batch:")
    print(f"   Neural: {neural.shape}")  # (8, neurons, 60)
    print(f"   Velocity: {velocity_seq.shape}")  # (8, 60)
    print(f"   Velocity range: [{velocity_seq.min():.3f}, {velocity_seq.max():.3f}]")
    
    # Check for peaks in sequence
    has_peaks = (torch.abs(velocity_seq) > 2.0).any(dim=1)
    print(f"   Samples with peaks (|v|>2): {has_peaks.sum().item()}/{len(has_peaks)}")
    
    print(f"\n✅ All tests passed!")