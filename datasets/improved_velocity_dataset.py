"""
Improved Velocity Dataset with Informative Neuron Selection

Seleziona i neuroni più correlati con la velocità per migliorare il training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.stats import pearsonr, spearmanr
from datasets.behavioral import load_behavioral_data_from_session
from datasets.calcium import extract_neural_data_from_hdf5


class InformativeVelocityDataset(Dataset):
    """
    Dataset migliorato che seleziona i neuroni più informativi per la velocità.
    
    Key improvements:
    - Seleziona top K neuroni basati sulla correlazione con velocità
    - Usa valori istantanei invece di medie
    - Opzione per augmentation temporale
    """
    
    def __init__(self, session_id, top_k_neurons=50, window_size=60, 
                 stride=15, normalize=True, selection_method='correlation',
                 use_instantaneous=True, temporal_offset=0):
        """
        Args:
            session_id: ID della sessione
            top_k_neurons: Numero di neuroni più informativi da selezionare
            window_size: Dimensione finestra temporale
            stride: Stride per sliding window (ridotto per più samples)
            normalize: Normalizza velocity
            selection_method: 'correlation', 'mutual_info', o 'variance'
            use_instantaneous: Se True usa valore istantaneo, altrimenti media
            temporal_offset: Offset temporale per predizione futura (default 0)
        """
        self.session_id = session_id
        self.top_k_neurons = top_k_neurons
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.selection_method = selection_method
        self.use_instantaneous = use_instantaneous
        self.temporal_offset = temporal_offset
        
        # Storage
        self.neural_windows = []
        self.velocity_windows = []
        self.selected_neurons = None
        
        # Carica e processa
        self._load_and_select_neurons()
        
        # Normalizza
        if self.normalize and len(self.velocity_windows) > 0:
            self._robust_normalize_velocity()
        
        print(f"\n✅ Informative Velocity Dataset created:")
        print(f"   Session: {session_id}")
        print(f"   Selected neurons: {len(self.selected_neurons)} (top {top_k_neurons})")
        print(f"   Total windows: {len(self.neural_windows)}")
        print(f"   Unique velocity values: {len(np.unique(self.velocity_windows))}")
    
    def _load_and_select_neurons(self):
        """Carica dati e seleziona neuroni informativi"""
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
        
        boc = BrainObservatoryCache()
        
        print(f"\n📊 Loading session {self.session_id}...")
        
        try:
            # Load neural data
            neural_result = extract_neural_data_from_hdf5(self.session_id)
            if neural_result is None or not neural_result['success']:
                raise RuntimeError(f"Failed to load neural data")
            
            dff_data = neural_result['dff_data']
            
            # Ensure (neurons, time)
            if dff_data.shape[0] > dff_data.shape[1]:
                dff_data = dff_data.T
            
            num_neurons = dff_data.shape[0]
            total_time = dff_data.shape[1]
            
            print(f"   Neural data: {dff_data.shape}")
            
            # Load behavioral data
            behavioral_data = load_behavioral_data_from_session(self.session_id, boc)
            if behavioral_data is None:
                raise RuntimeError(f"Failed to load behavioral data")
            
            velocity_data = behavioral_data['running_speed']
            
            # Align temporal dimensions
            min_time = min(total_time, len(velocity_data))
            dff_data = dff_data[:, :min_time]
            velocity_data = velocity_data[:min_time]
            
            # ========================================================================
            # SELEZIONA NEURONI PIÙ INFORMATIVI
            # ========================================================================
            print(f"\n🔍 Selecting informative neurons using {self.selection_method}...")
            
            if self.selection_method == 'correlation':
                # Calcola correlazione di ogni neurone con velocità
                correlations = []
                for i in range(num_neurons):
                    # Usa solo i punti validi (non NaN)
                    valid_mask = ~(np.isnan(dff_data[i]) | np.isnan(velocity_data))
                    if np.sum(valid_mask) > 100:
                        corr, _ = pearsonr(dff_data[i][valid_mask], velocity_data[valid_mask])
                        correlations.append(abs(corr))  # Usa valore assoluto
                    else:
                        correlations.append(0)
                
                correlations = np.array(correlations)
                selected_indices = np.argsort(correlations)[-self.top_k_neurons:][::-1]
                
                print(f"   Top correlations: {correlations[selected_indices[:5]]}")
                print(f"   Mean correlation (selected): {correlations[selected_indices].mean():.4f}")
                
            elif self.selection_method == 'variance':
                # Seleziona neuroni con maggiore varianza
                variances = np.var(dff_data, axis=1)
                selected_indices = np.argsort(variances)[-self.top_k_neurons:][::-1]
                
                print(f"   Top variances: {variances[selected_indices[:5]]}")
                
            elif self.selection_method == 'mutual_info':
                # Mutual information (più costoso computazionalmente)
                from sklearn.feature_selection import mutual_info_regression
                
                # Downsample per velocità
                downsample = 10
                X = dff_data[:, ::downsample].T  # (time, neurons)
                y = velocity_data[::downsample]
                
                mi_scores = mutual_info_regression(X, y)
                selected_indices = np.argsort(mi_scores)[-self.top_k_neurons:][::-1]
                
                print(f"   Top MI scores: {mi_scores[selected_indices[:5]]}")
            
            else:
                # Random selection as baseline
                selected_indices = np.random.choice(num_neurons, self.top_k_neurons, replace=False)
            
            self.selected_neurons = selected_indices
            selected_dff = dff_data[selected_indices]
            
            print(f"   ✅ Selected {len(selected_indices)} neurons")
            
            # ========================================================================
            # CREA WINDOWS CON DIVERSIFICAZIONE
            # ========================================================================
            self._create_diverse_windows(selected_dff, velocity_data)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_diverse_windows(self, neural_data, velocity_data):
        """
        Crea windows con target più diversificati
        """
        total_time = neural_data.shape[1]
        
        for start in range(0, total_time - self.window_size - self.temporal_offset + 1, self.stride):
            end = start + self.window_size
            
            neural_window = neural_data[:, start:end]
            
            if self.use_instantaneous:
                # Usa valore istantaneo con offset temporale
                target_idx = min(end + self.temporal_offset - 1, len(velocity_data) - 1)
                velocity_value = velocity_data[target_idx]
            else:
                # Usa statistiche più complesse
                window_velocities = velocity_data[start:end]
                
                # Opzioni per target più informativi:
                # 1. Max assoluto (cattura picchi)
                # velocity_value = np.max(np.abs(window_velocities)) * np.sign(window_velocities[-1])
                
                # 2. Differenza (cattura accelerazione)
                # velocity_value = window_velocities[-1] - window_velocities[0]
                
                # 3. Media pesata (più peso ai valori recenti)
                weights = np.exp(np.linspace(-1, 0, len(window_velocities)))
                weights /= weights.sum()
                velocity_value = np.average(window_velocities, weights=weights)
            
            # Solo aggiungi se il valore non è NaN
            if not np.isnan(velocity_value):
                self.neural_windows.append(neural_window)
                self.velocity_windows.append(velocity_value)
    
    def _robust_normalize_velocity(self):
        """Normalizzazione robusta con clipping basato su percentili"""
        velocity_array = np.array(self.velocity_windows)
        
        print(f"\n📊 Robust normalization:")
        print(f"   Original - Min: {velocity_array.min():.4f}, Max: {velocity_array.max():.4f}")
        print(f"   Unique values: {len(np.unique(velocity_array))}")
        
        # Rimuovi outliers estremi usando IQR
        q1 = np.percentile(velocity_array, 25)
        q3 = np.percentile(velocity_array, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        velocity_clipped = np.clip(velocity_array, lower_bound, upper_bound)
        
        # Z-score normalization
        self.velocity_mean = np.mean(velocity_clipped)
        self.velocity_std = np.std(velocity_clipped)
        
        if self.velocity_std < 1e-8:
            self.velocity_std = 1.0
        
        velocity_norm = (velocity_clipped - self.velocity_mean) / self.velocity_std
        
        self.velocity_windows = velocity_norm.tolist()
        
        print(f"   After normalization - Min: {velocity_norm.min():.4f}, Max: {velocity_norm.max():.4f}")
        print(f"   Mean: {velocity_norm.mean():.6f}, Std: {velocity_norm.std():.6f}")
        print(f"   Unique normalized values: {len(np.unique(velocity_norm))}")
    
    def __len__(self):
        return len(self.neural_windows)
    
    def __getitem__(self, idx):
        neural = torch.FloatTensor(self.neural_windows[idx])
        velocity = torch.FloatTensor([self.velocity_windows[idx]]).squeeze()
        return neural, velocity


def create_informative_dataloaders(session_id, top_k_neurons=50, test_split=0.2,
                                   stride=15, selection_method='correlation',
                                   batch_size=32, num_workers=0):
    """
    Factory function per creare dataloaders con selezione neuroni informativi
    """
    
    print("="*70)
    print(f"🎯 CREATING INFORMATIVE VELOCITY DATALOADERS")
    print(f"   Strategy: Select top {top_k_neurons} most informative neurons")
    print(f"   Method: {selection_method}")
    print("="*70)
    
    # Create dataset
    dataset = InformativeVelocityDataset(
        session_id=session_id,
        top_k_neurons=top_k_neurons,
        window_size=60,
        stride=stride,
        normalize=True,
        selection_method=selection_method,
        use_instantaneous=True,
        temporal_offset=0
    )
    
    # Split
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders with proper batch size
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
    
    dataset_info = {
        'session_id': session_id,
        'total_samples': total_size,
        'train_samples': train_size,
        'test_samples': test_size,
        'num_neurons': top_k_neurons,
        'selected_neurons': dataset.selected_neurons.tolist(),
        'unique_targets': len(np.unique(dataset.velocity_windows)),
        'normalized': True
    }
    
    print(f"\n✅ Dataloaders created!")
    print(f"   Train: {train_size} windows")
    print(f"   Test: {test_size} windows")
    print(f"   Unique velocity targets: {dataset_info['unique_targets']}")
    print("="*70)
    
    return train_loader, test_loader, dataset_info


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing InformativeVelocityDataset\n")
    
    from datasets.calcium import TRAINING_SESSION_IDS
    
    session_id = TRAINING_SESSION_IDS[0]
    
    # Test different selection methods
    for method in ['correlation', 'variance']:
        print(f"\n📊 Testing {method} selection...")
        
        train_loader, test_loader, info = create_informative_dataloaders(
            session_id=session_id,
            top_k_neurons=50,
            selection_method=method,
            batch_size=32,
            stride=10  # Più samples
        )
        
        print(f"   Unique targets: {info['unique_targets']}")
        
        # Check first batch
        batch = next(iter(train_loader))
        neural, velocity = batch
        
        print(f"   Batch shapes - Neural: {neural.shape}, Velocity: {velocity.shape}")
        print(f"   Velocity range: [{velocity.min():.4f}, {velocity.max():.4f}]")
        print(f"   Velocity std: {velocity.std():.4f}")
    
    print("\n✅ All tests passed!")