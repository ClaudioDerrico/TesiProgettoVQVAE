"""
Allen Brain Observatory Behavioral Dataset - FIXED VERSION

CAMBIAMENTO PRINCIPALE:
- USA API AllenSDK (dataset.get_running_speed(), etc.) invece di accesso diretto HDF5
- Questo risolve il problema "No pupil data found" / "No running speed found"

Loads and aligns:
- Neural calcium imaging data
- Behavioral data:
  * Pupil dilation
  * Eye tracking (vertical, horizontal position)
  * Running speed (velocity)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def load_behavioral_data_from_session(session_id, boc=None):
    """
    Carica dati comportamentali da una sessione Allen Brain Observatory
    USANDO L'API AllenSDK (non accesso diretto HDF5)
    
    Args:
        session_id: Session ID
        boc: BrainObservatoryCache instance (optional)
    
    Returns:
        dict with:
        - 'pupil': (time,) pupil area
        - 'eye_vert': (time,) vertical eye position  
        - 'eye_horiz': (time,) horizontal eye position
        - 'running_speed': (time,) running velocity
        - 'timestamps': (time,) timestamps (matching neural data)
    """
    
    if boc is None:
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
        boc = BrainObservatoryCache()
    
    print(f"📊 Loading behavioral data for session {session_id}...")
    
    try:
        # Get dataset
        dataset = boc.get_ophys_experiment_data(session_id)
        
        # ====================================================================
        # GET NEURAL TIMESTAMPS (reference) - PRIMA DI TUTTO
        # ====================================================================
        with h5py.File(dataset.nwb_file, 'r') as f:
            neural_path = 'processing/brain_observatory_pipeline/DfOverF/imaging_plane_1/timestamps'
            neural_timestamps = f[neural_path][:]
        
        print(f"   📐 Neural timestamps: {len(neural_timestamps)} samples")
        
        # ====================================================================
        # RUNNING SPEED - USA API AllenSDK
        # ====================================================================
        running_speed = None
        running_timestamps = None
        
        try:
            running_data = dataset.get_running_speed()
            
            if running_data is not None and len(running_data) == 2:
                running_speed, running_timestamps = running_data
                
                # Remove NaNs
                valid_mask = ~np.isnan(running_speed)
                valid_count = np.sum(valid_mask)
                
                if valid_count > 100 and np.std(running_speed[valid_mask]) > 0.1:
                    print(f"   ✅ Running speed: {len(running_speed)} samples ({100*valid_count/len(running_speed):.1f}% valid)")
                else:
                    print(f"   ⚠️ Running speed data has low variance or too many NaNs")
                    running_speed = None
                    running_timestamps = None
            else:
                print(f"   ⚠️ No running speed found")
                
        except Exception as e:
            print(f"   ⚠️ No running speed: {e}")
        
        # ====================================================================
        # PUPIL SIZE - USA API AllenSDK
        # ====================================================================
        pupil_area = None
        pupil_timestamps = None
        
        try:
            pupil_size = dataset.get_pupil_size()
            
            if pupil_size is not None and len(pupil_size) == 2:
                pupil_data, pupil_timestamps = pupil_size
                pupil_area = np.array(pupil_data)
                
                # Remove NaNs
                valid_mask = ~np.isnan(pupil_area)
                valid_count = np.sum(valid_mask)
                
                if valid_count > 100 and np.std(pupil_area[valid_mask]) > 1.0:
                    print(f"   ✅ Pupil size: {len(pupil_area)} samples ({100*valid_count/len(pupil_area):.1f}% valid)")
                else:
                    print(f"   ⚠️ Pupil data has low variance or too many NaNs")
                    pupil_area = None
                    pupil_timestamps = None
            else:
                print(f"   ⚠️ No pupil data found")
                
        except Exception as e:
            print(f"   ⚠️ No pupil data: {e}")
        
        # ====================================================================
        # EYE POSITION - USA API AllenSDK
        # ====================================================================
        eye_x = None
        eye_y = None
        eye_timestamps = None
        
        try:
            pupil_location = dataset.get_pupil_location()
            
            if pupil_location is not None:
                # Check if it's array-like with proper shape
                if hasattr(pupil_location, 'shape') and len(pupil_location.shape) >= 2:
                    if pupil_location.shape[1] >= 2:
                        eye_x = pupil_location[:, 0]
                        eye_y = pupil_location[:, 1]
                        
                        # Use same timestamps as pupil (or estimate)
                        if pupil_timestamps is not None:
                            eye_timestamps = pupil_timestamps
                        else:
                            # Estimate timestamps
                            eye_timestamps = np.linspace(
                                neural_timestamps[0], 
                                neural_timestamps[-1], 
                                len(eye_x)
                            )
                        
                        # Check validity
                        valid_x = ~np.isnan(eye_x)
                        valid_y = ~np.isnan(eye_y)
                        valid_both = valid_x & valid_y
                        valid_count = np.sum(valid_both)
                        
                        if valid_count > 100:
                            x_std = np.std(eye_x[valid_both])
                            y_std = np.std(eye_y[valid_both])
                            
                            if x_std > 1.0 or y_std > 1.0:
                                print(f"   ✅ Eye position: {len(eye_x)} samples ({100*valid_count/len(eye_x):.1f}% valid)")
                            else:
                                print(f"   ⚠️ Eye position has low variance")
                                eye_x = None
                                eye_y = None
                                eye_timestamps = None
                        else:
                            print(f"   ⚠️ Eye position has too many NaNs")
                            eye_x = None
                            eye_y = None
                            eye_timestamps = None
                else:
                    print(f"   ⚠️ Eye position unexpected shape: {pupil_location.shape}")
            else:
                print(f"   ⚠️ No eye position found")
                
        except Exception as e:
            print(f"   ⚠️ No eye position: {e}")
        
        # ====================================================================
        # TEMPORAL ALIGNMENT
        # ====================================================================
        aligned_behavioral = {}
        
        # Running speed
        if running_speed is not None and running_timestamps is not None:
            aligned_behavioral['running_speed'] = interpolate_to_neural_timestamps(
                running_speed, running_timestamps, neural_timestamps
            )
        else:
            aligned_behavioral['running_speed'] = np.zeros_like(neural_timestamps)
        
        # Pupil area
        if pupil_area is not None and pupil_timestamps is not None:
            aligned_behavioral['pupil'] = interpolate_to_neural_timestamps(
                pupil_area, pupil_timestamps, neural_timestamps
            )
        else:
            aligned_behavioral['pupil'] = np.zeros_like(neural_timestamps)
        
        # Eye position
        if eye_x is not None and eye_y is not None and eye_timestamps is not None:
            aligned_behavioral['eye_horiz'] = interpolate_to_neural_timestamps(
                eye_x, eye_timestamps, neural_timestamps
            )
            aligned_behavioral['eye_vert'] = interpolate_to_neural_timestamps(
                eye_y, eye_timestamps, neural_timestamps
            )
        else:
            aligned_behavioral['eye_horiz'] = np.zeros_like(neural_timestamps)
            aligned_behavioral['eye_vert'] = np.zeros_like(neural_timestamps)
        
        aligned_behavioral['timestamps'] = neural_timestamps
        
        print(f"   ✅ Behavioral data aligned to {len(neural_timestamps)} neural timestamps")
        
        return aligned_behavioral
        
    except Exception as e:
        print(f"   ❌ Failed to load behavioral data: {e}")
        import traceback
        traceback.print_exc()
        return None


def interpolate_to_neural_timestamps(behavioral_data, behavioral_timestamps, neural_timestamps):
    """
    Interpolate behavioral data to match neural timestamps
    
    Args:
        behavioral_data: (behavioral_time,)
        behavioral_timestamps: (behavioral_time,)
        neural_timestamps: (neural_time,)
    
    Returns:
        interpolated_data: (neural_time,)
    """
    
    # Remove NaNs
    valid_mask = ~np.isnan(behavioral_data)
    
    if np.sum(valid_mask) < 2:
        # Not enough valid data, return zeros
        return np.zeros_like(neural_timestamps)
    
    behavioral_data_clean = behavioral_data[valid_mask]
    behavioral_timestamps_clean = behavioral_timestamps[valid_mask]
    
    # Interpolate
    interpolator = interp1d(
        behavioral_timestamps_clean,
        behavioral_data_clean,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    
    interpolated = interpolator(neural_timestamps)
    
    # Smooth (optional)
    interpolated = gaussian_filter1d(interpolated, sigma=1.0)
    
    return interpolated


class AllenBehavioralDataset(Dataset):
    """
    Dataset con neural + behavioral data da Allen Brain Observatory
    
    Struttura:
    - neural_data: (num_neurons, window_size)
    - behavioral_data: (4,) = [pupil, eye_vert, eye_horiz, running_speed]
    
    Ogni sample è una finestra temporale con:
    - Neural activity (input)
    - Behavioral variables (target)
    """
    
    def __init__(self, session_ids, window_size=60, stride=30, 
                 num_neurons=1, normalize=True):
        """
        Args:
            session_ids: Lista di session IDs da caricare
            window_size: Dimensione finestra temporale
            stride: Stride per sliding window
            num_neurons: Numero di neuroni da usare (1 per single-neuron)
            normalize: Se True, normalizza behavioral data
        """
        
        self.window_size = window_size
        self.stride = stride
        self.num_neurons = num_neurons
        self.normalize = normalize
        
        # Storage
        self.neural_windows = []
        self.behavioral_windows = []
        
        # Carica tutte le sessioni
        self._load_all_sessions(session_ids)
        
        # Normalizza se richiesto
        if self.normalize:
            self._normalize_behavioral_data()
        
        print(f"\n✅ Dataset creato:")
        print(f"   Total windows: {len(self.neural_windows)}")
        print(f"   Neural shape per window: ({self.num_neurons}, {self.window_size})")
        print(f"   Behavioral shape: (4,)")
    
    def _load_all_sessions(self, session_ids):
        """Carica tutte le sessioni"""
        
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
        from datasets.calcium import extract_neural_data_from_hdf5, select_random_neurons
        
        boc = BrainObservatoryCache()
        
        print(f"\n📊 Loading {len(session_ids)} sessions...")
        
        for i, session_id in enumerate(session_ids, 1):
            print(f"\n[{i}/{len(session_ids)}] Session {session_id}")
            
            try:
                # ============================================================
                # NEURAL DATA
                # ============================================================
                neural_result = extract_neural_data_from_hdf5(session_id)
                
                if neural_result is None or not neural_result['success']:
                    print(f"   ❌ Failed to load neural data")
                    continue
                
                dff_data = neural_result['dff_data']
                
                # Assicura (neurons, time)
                if dff_data.shape[0] > dff_data.shape[1]:
                    dff_data = dff_data.T
                
                # Seleziona neuroni
                if dff_data.shape[0] > self.num_neurons:
                    selected_indices = select_random_neurons(dff_data, self.num_neurons)
                    dff_data = dff_data[selected_indices]
                
                print(f"   Neural: {dff_data.shape}")
                
                # ============================================================
                # BEHAVIORAL DATA
                # ============================================================
                behavioral_data = load_behavioral_data_from_session(session_id, boc)
                
                if behavioral_data is None:
                    print(f"   ❌ Failed to load behavioral data")
                    continue
                
                # Verifica allineamento temporale
                neural_time = dff_data.shape[1]
                behavioral_time = len(behavioral_data['timestamps'])
                
                if neural_time != behavioral_time:
                    print(f"   ⚠️ Mismatch: neural={neural_time}, behavioral={behavioral_time}")
                    # Tronca al minimo
                    min_time = min(neural_time, behavioral_time)
                    dff_data = dff_data[:, :min_time]
                    for key in ['pupil', 'eye_vert', 'eye_horiz', 'running_speed']:
                        behavioral_data[key] = behavioral_data[key][:min_time]
                
                # ============================================================
                # CREATE WINDOWS
                # ============================================================
                self._create_windows_from_session(dff_data, behavioral_data)
                
                print(f"   ✅ Loaded {len(self.neural_windows)} total windows so far")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    def _create_windows_from_session(self, neural_data, behavioral_data):
        """
        Crea sliding windows da una sessione
        
        Args:
            neural_data: (neurons, time)
            behavioral_data: dict with behavioral variables
        """
        
        total_time = neural_data.shape[1]
        
        # Sliding window
        for start in range(0, total_time - self.window_size + 1, self.stride):
            end = start + self.window_size
            
            # Neural window
            neural_window = neural_data[:, start:end]
            
            # Behavioral window: usa valore MEDIO della finestra
            behavioral_window = np.array([
                np.mean(behavioral_data['pupil'][start:end]),
                np.mean(behavioral_data['eye_vert'][start:end]),
                np.mean(behavioral_data['eye_horiz'][start:end]),
                np.mean(behavioral_data['running_speed'][start:end])
            ])
            
            self.neural_windows.append(neural_window)
            self.behavioral_windows.append(behavioral_window)
    
    def _normalize_behavioral_data(self):
        """Normalizza behavioral data (Z-score)"""
        
        if len(self.behavioral_windows) == 0:
            return
        
        # Stack
        behavioral_array = np.stack(self.behavioral_windows, axis=0)  # (N, 4)
        
        # Calcola mean e std per variabile
        self.behavioral_mean = np.mean(behavioral_array, axis=0)
        self.behavioral_std = np.std(behavioral_array, axis=0)
        
        # Avoid division by zero
        self.behavioral_std = np.where(self.behavioral_std < 1e-8, 1.0, self.behavioral_std)
        
        # Normalizza
        behavioral_array_norm = (behavioral_array - self.behavioral_mean) / self.behavioral_std
        
        # Update
        self.behavioral_windows = [behavioral_array_norm[i] for i in range(len(self.behavioral_windows))]
        
        print(f"\n   📊 Behavioral normalization:")
        var_names = ['pupil', 'eye_vert', 'eye_horiz', 'running_speed']
        for i, name in enumerate(var_names):
            print(f"      {name}: mean={self.behavioral_mean[i]:.3f}, std={self.behavioral_std[i]:.3f}")
    
    def __len__(self):
        return len(self.neural_windows)
    
    def __getitem__(self, idx):
        """
        Returns:
            neural_data: (num_neurons, window_size)
            behavioral_data: (4,)
        """
        neural = torch.FloatTensor(self.neural_windows[idx])
        behavioral = torch.FloatTensor(self.behavioral_windows[idx])
        
        return neural, behavioral


def create_behavioral_dataloaders(session_ids, batch_size=32, test_split=0.2,
                                  window_size=60, stride=30, num_neurons=1,
                                  num_workers=0, normalize=True):
    """
    Factory function per creare dataloaders con behavioral data
    
    Args:
        session_ids: Lista session IDs
        batch_size: Batch size
        test_split: Frazione per test
        window_size: Window size
        stride: Stride
        num_neurons: Numero neuroni
        num_workers: Workers
        normalize: Normalizza behavioral data
    
    Returns:
        train_loader, test_loader, dataset_info
    """
    
    print("="*70)
    print("🧠 CREATING BEHAVIORAL DATALOADERS")
    print("="*70)
    
    # Create dataset
    dataset = AllenBehavioralDataset(
        session_ids=session_ids,
        window_size=window_size,
        stride=stride,
        num_neurons=num_neurons,
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
        num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    dataset_info = {
        'total_samples': total_size,
        'train_samples': train_size,
        'test_samples': test_size,
        'num_sessions': len(session_ids),
        'neural_shape': (num_neurons, window_size),
        'behavioral_shape': (4,),
        'normalized': normalize
    }
    
    print(f"\n✅ Dataloaders created!")
    print(f"   Train: {train_size}")
    print(f"   Test: {test_size}")
    print("="*70)
    
    return train_loader, test_loader, dataset_info