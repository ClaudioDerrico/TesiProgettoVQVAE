import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, zscore
from sklearn.preprocessing import StandardScaler, RobustScaler
import h5py
import gc
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURAZIONE SESSIONI
# ============================================================================

# 10 NUOVE SESSIONI per TRAINING (VISp, mai usate prima)
TRAINING_SESSION_IDS = [
    501704220,  # VISp
    501271265,  # VISp
    502205092,  # VISp
    501773889,  # VISp
    502066273,  # VISp
    502115959,  # VISp
    501950670,  # VISp
    502376417,  # VISp
    503109347,  # VISp
    502608215,  # VISp
]

# 3 SESSIONI per TEST CROSS-SESSION (quelle già usate prima)
TEST_SESSION_IDS = [
    501559087,  # VISp, Slc17a7-IRES2-Cre
    501498760,  # VISp
    501836392,  # VISp
]

# SESSIONE ORIGINALE (ora deprecata, era usata prima per training)
ORIGINAL_TRAINING_SESSION = 501474098

print(f"📋 Configurate {len(TRAINING_SESSION_IDS)} sessioni per TRAINING")
print(f"📋 Configurate {len(TEST_SESSION_IDS)} sessioni per TEST CROSS-SESSION")


class CalciumDataset(Dataset):
    """
    Simplified dataset for calcium imaging data focused on reconstruction.
    
    ✅ REMOVED: behavior data, augmentation, complex preprocessing
    ✅ FOCUSED: Only neural data for VQ-VAE reconstruction task
    """
    
    def __init__(self, neural_data, augment=False, augment_prob=0.3, noise_std=0.05):
        """
        Args:
            neural_data: numpy array (n_samples, n_neurons, time_steps)
            augment: whether to apply minimal augmentation
            augment_prob: probability of applying augmentation
            noise_std: standard deviation for gaussian noise augmentation
        """
        self.neural_data = torch.FloatTensor(neural_data)
        self.augment = augment
        self.augment_prob = augment_prob
        self.noise_std = noise_std
        
        print(f"📊 CalciumDataset created:")
        print(f"   Samples: {len(self)}")
        print(f"   Neural data shape: {self.neural_data.shape}")
        print(f"   Augmentation: {'Enabled' if augment else 'Disabled'}")
        
    def __len__(self):
        return len(self.neural_data)
    
    def __getitem__(self, idx):
        neural = self.neural_data[idx].clone()
        
        # Apply minimal augmentation during training
        if self.augment and np.random.rand() < self.augment_prob:
            neural = self._augment_neural_data(neural)
            
        return neural
    
    def _augment_neural_data(self, neural_data):
        """Apply minimal augmentation techniques to neural data."""
        
        # 1. Gaussian noise (ridotto)
        if np.random.rand() < 0.4:
            noise = torch.randn_like(neural_data) * self.noise_std
            neural_data = neural_data + noise
        
        # 2. Neuron dropout (molto ridotto)
        if np.random.rand() < 0.05:
            dropout_prob = np.random.uniform(0.02, 0.08)
            dropout_mask = torch.rand(neural_data.shape[0]) > dropout_prob
            neural_data = neural_data * dropout_mask.unsqueeze(-1)
        
        return neural_data


def extract_speed_from_hdf5(session_id):
    """Extract running speed directly from HDF5/NWB file."""
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    
    print(f"🔍 Extracting data directly from HDF5 for session {session_id}...")
    
    boc = BrainObservatoryCache()
    
    try:
        ds = boc.get_ophys_experiment_data(session_id)
        fpath = ds.nwb_file
        print(f"   📁 NWB file path: {fpath}")
        
        del ds
        gc.collect()
        
        with h5py.File(fpath, 'r', rdcc_nbytes=0) as f:
            
            B = 'processing/brain_observatory_pipeline'
            speed_path = f'{B}/BehavioralTimeSeries/running_speed/data'
            speed_ts_path = f'{B}/BehavioralTimeSeries/running_speed/timestamps'
            dff_path = f'{B}/DfOverF/imaging_plane_1/data'
            dff_ts_path = f'{B}/DfOverF/imaging_plane_1/timestamps'
            
            if speed_path in f and speed_ts_path in f:
                print("   ✅ Found running speed data in HDF5")
                
                speed_data = f[speed_path][:]
                speed_timestamps = f[speed_ts_path][:]
                
                print(f"   📊 Speed data shape: {speed_data.shape}")
                print(f"   📈 Speed range: {np.nanmin(speed_data):.3f} - {np.nanmax(speed_data):.3f}")
                
                nan_count = np.sum(np.isnan(speed_data))
                valid_pct = 100 * (len(speed_data) - nan_count) / len(speed_data)
                print(f"   ✅ Valid speed data: {valid_pct:.1f}% ({len(speed_data) - nan_count}/{len(speed_data)})")
                
                speed_range = np.nanmax(speed_data) - np.nanmin(speed_data)
                realistic_speed = (0.1 < speed_range < 100)
                speed_valid = realistic_speed and valid_pct > 50
                
            else:
                print("   ❌ Running speed data not found in HDF5")
                speed_data, speed_timestamps = None, None
                speed_valid = False
            
            if dff_path in f and dff_ts_path in f:
                print("   ✅ Found neural data in HDF5")
                
                dff_data = f[dff_path][:]
                dff_timestamps = f[dff_ts_path][:]
                
                if dff_data.ndim == 2 and dff_data.shape[0] != dff_timestamps.shape[0]:
                    dff_data = dff_data.T
                    print("   🔄 Transposed DFF data for correct alignment")
                
                print(f"   📊 DFF data shape: {dff_data.shape}")
                
            else:
                print("   ❌ Neural data not found in HDF5")
                dff_data, dff_timestamps = None, None
        
        extraction_result = {
            'speed_data': speed_data,
            'speed_timestamps': speed_timestamps,
            'dff_data': dff_data,
            'dff_timestamps': dff_timestamps,
            'speed_valid': speed_valid,
            'session_id': session_id,
            'extraction_method': 'direct_hdf5'
        }
        
        return extraction_result
        
    except Exception as e:
        print(f"   ❌ HDF5 extraction failed: {e}")
        return None


def preprocess_neural_data_only(extraction_result):
    """Preprocessa solo i dati neurali, speed solo per selezione neuroni"""
    if extraction_result is None:
        print("   ⚠️ No valid data available")
        return None
    
    print("   🔧 Preprocessing neural data...")
    
    dff_data = extraction_result['dff_data']
    dff_timestamps = extraction_result['dff_timestamps']
    speed_data = extraction_result.get('speed_data')
    speed_timestamps = extraction_result.get('speed_timestamps')
    
    if speed_data is not None and speed_timestamps is not None and extraction_result['speed_valid']:
        speed_aligned = np.interp(dff_timestamps, speed_timestamps, speed_data, 
                                 left=speed_data[0], right=speed_data[-1])
        speed_smooth = gaussian_filter1d(speed_aligned, sigma=5)
    else:
        print("   ⚠️ No valid speed data - using dummy values for neuron selection")
        speed_smooth = np.random.randn(len(dff_timestamps)) * 0.1
    
    return {
        'timestamps': dff_timestamps,
        'dff_traces': dff_data,
        'speed_for_selection': speed_smooth,
        'metadata': {
            'session_id': extraction_result['session_id'],
            'num_neurons': dff_data.shape[1] if dff_data.ndim == 2 else dff_data.shape[0],
            'num_timepoints': dff_data.shape[0] if dff_data.ndim == 2 else dff_data.shape[1],
            'duration_seconds': dff_timestamps[-1] - dff_timestamps[0],
        }
    }


class SimpleAllenBrainDataset(Dataset):
    """
    Allen Brain Observatory dataset focused only on neural data reconstruction.
    Supporta sia singola sessione che multi-sessione.
    
    NUOVI PARAMETRI:
    - window_size = 60 timesteps (era 50)
    - stride = 50 timesteps (era 10)
    """
    
    def __init__(self, window_size=60, stride=50, min_neurons=30, augment=False, session_id=None):
        """
        Args:
            window_size: size of temporal windows (DEFAULT: 60 timesteps)
            stride: stride for sliding windows (DEFAULT: 50 timesteps)
            min_neurons: minimum number of neurons to keep (30)
            augment: whether to apply minimal augmentation
            session_id: specific session ID or None for default
        """
        
        if session_id is None:
            self.session_id = TRAINING_SESSION_IDS[0]  # Default: prima sessione training
        else:
            self.session_id = session_id
    
        self.window_size = window_size
        self.stride = stride
        self.min_neurons = min_neurons
        self.augment = augment
        
        # Load and preprocess data
        self._load_allen_data()
        
    def _load_allen_data(self):
        """Load data focusing only on neural reconstruction."""
        print("🧠 Loading Allen Brain Observatory data for reconstruction...")
        print(f"Using session: {self.session_id}")
        print(f"Parameters: window_size={self.window_size}, stride={self.stride}")
        
        extraction_result = extract_speed_from_hdf5(self.session_id)
        
        if extraction_result:
            processed_data = preprocess_neural_data_only(extraction_result)
            
            if processed_data:
                print("  ✅ Successfully extracted neural data from HDF5!")
                
                timestamps = processed_data['timestamps']
                dff_traces = processed_data['dff_traces']
                speed_for_selection = processed_data['speed_for_selection']
                
                if dff_traces.ndim == 2 and dff_traces.shape[0] > dff_traces.shape[1]:
                    dff_traces = dff_traces.T
                    print("  🔄 Transposed DFF traces to (neurons, time)")
                
                print(f"  📊 Final neural data shape: {dff_traces.shape}")
                
            else:
                print("  ⚠️ HDF5 preprocessing failed - falling back to Allen SDK")
                timestamps, dff_traces, speed_for_selection = self._fallback_to_allen_sdk()
        else:
            print("  ⚠️ HDF5 extraction failed - falling back to Allen SDK")
            timestamps, dff_traces, speed_for_selection = self._fallback_to_allen_sdk()
        
        neural_windows = self._preprocess_neural_only(dff_traces, speed_for_selection)
        
        self.neural_data = torch.FloatTensor(neural_windows)
        
        print(f"📊 Final dataset: {len(self)} windows")
        print(f"   Neural data shape: {self.neural_data.shape}")
    
    def _fallback_to_allen_sdk(self):
        """Fallback method using standard Allen SDK API."""
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
        
        print("  🔄 Using Allen SDK API as fallback...")
        boc = BrainObservatoryCache()
        data_set = boc.get_ophys_experiment_data(self.session_id)
        
        timestamps, dff_traces = data_set.get_dff_traces()
        run_ts, running_speed = data_set.get_running_speed()
        
        print(f"  📊 Allen SDK data shapes:")
        print(f"     DFF traces: {dff_traces.shape}")
        print(f"     Running speed: {running_speed.shape}")
        
        if not np.array_equal(timestamps, run_ts):
            valid_mask = ~np.isnan(run_ts) & ~np.isnan(running_speed)
            if np.sum(valid_mask) > 0:
                speed_interp = interp1d(run_ts[valid_mask], running_speed[valid_mask], 
                                       bounds_error=False, fill_value=np.nanmean(running_speed))
                speed_aligned = speed_interp(timestamps)
            else:
                speed_aligned = np.random.randn(len(timestamps)) * 0.1
        else:
            speed_aligned = running_speed
        
        speed_for_selection = gaussian_filter1d(speed_aligned, sigma=5)
        
        return timestamps, dff_traces, speed_for_selection
    
    def _preprocess_neural_only(self, dff_traces, speed_for_selection):
        """✅ SIMPLIFIED: Preprocess only neural data."""
        
        # Select active neurons
        active_neurons = self._select_active_neurons(dff_traces, speed_for_selection)
        dff_active = dff_traces[active_neurons, :]
        
        print(f"  ✅ Selected {np.sum(active_neurons)} active neurons")
        
        # Normalize neural data
        dff_normalized = zscore(dff_active, axis=1)
        dff_normalized = np.nan_to_num(dff_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create temporal windows
        neural_windows = self._create_neural_windows(dff_normalized)
        
        return neural_windows
    
    def _select_active_neurons(self, dff_traces, speed_data):
        """Select neurons based on activity and correlation with behavior."""
        
        correlations = []
        variances = np.var(dff_traces, axis=1)
        
        for i, neuron_trace in enumerate(dff_traces):
            trace_clean = np.nan_to_num(neuron_trace, nan=0.0)
            speed_clean = np.nan_to_num(speed_data, nan=0.0)
            
            if np.std(trace_clean) > 1e-8 and np.std(speed_clean) > 1e-8:
                try:
                    corr, _ = pearsonr(trace_clean, speed_clean)
                    correlations.append(abs(corr) if np.isfinite(corr) else 0)
                except:
                    correlations.append(0)
            else:
                correlations.append(0)
        
        correlations = np.array(correlations)
        
        # Combined score: variance + correlation
        variance_rank = np.argsort(variances)
        correlation_rank = np.argsort(correlations)
        
        combined_score = np.zeros(len(dff_traces))
        for i in range(len(dff_traces)):
            var_percentile = np.where(variance_rank == i)[0][0] / len(variance_rank)
            corr_percentile = np.where(correlation_rank == i)[0][0] / len(correlation_rank)
            combined_score[i] = 0.7 * var_percentile + 0.3 * corr_percentile
        
        # Select top neurons
        top_neurons = np.argsort(combined_score)[-self.min_neurons:]
        active_neurons = np.zeros(len(dff_traces), dtype=bool)
        active_neurons[top_neurons] = True
        
        print(f"  📊 Neuron selection stats:")
        print(f"     Mean correlation: {np.mean(correlations[active_neurons]):.3f}")
        print(f"     Mean variance: {np.mean(variances[active_neurons]):.3f}")
        
        return active_neurons
    
    def _create_neural_windows(self, neural_data):
        """Create sliding temporal windows with NEW stride"""
        neural_windows = []
        
        for start in range(0, neural_data.shape[1] - self.window_size + 1, self.stride):
            neural_window = neural_data[:, start:start + self.window_size]
            neural_windows.append(neural_window)
        
        return np.array(neural_windows)
    
    def __len__(self):
        return len(self.neural_data)
    
    def __getitem__(self, idx):
        neural = self.neural_data[idx]
        
        if self.augment:
            neural = self._augment_neural_data(neural)
        
        return neural
    
    def _augment_neural_data(self, neural_data):
        """Apply minimal augmentation specific to Allen Brain data."""
        if np.random.rand() < 0.3:
            noise_std = 0.03 * torch.std(neural_data)
            noise = torch.randn_like(neural_data) * noise_std
            neural_data = neural_data + noise
        
        return neural_data


def create_multi_session_dataset(session_ids, window_size=60, stride=50, min_neurons=30):
    """
    Crea un dataset combinato da multiple sessioni.
    
    Args:
        session_ids: lista di session ID da caricare
        window_size: dimensione finestre temporali (DEFAULT: 60)
        stride: stride per finestre (DEFAULT: 50)
        min_neurons: numero minimo neuroni (30)
    
    Returns:
        ConcatDataset contenente tutti i dati dalle sessioni
    """
    print(f"\n🎯 Creating MULTI-SESSION dataset from {len(session_ids)} sessions")
    print(f"   Window size: {window_size}, Stride: {stride}, Min neurons: {min_neurons}")
    print("="*70)
    
    datasets = []
    
    for i, session_id in enumerate(session_ids):
        print(f"\n📊 Loading session {i+1}/{len(session_ids)}: {session_id}")
        print("-"*70)
        
        try:
            dataset = SimpleAllenBrainDataset(
                window_size=window_size,
                stride=stride,
                min_neurons=min_neurons,
                session_id=session_id
            )
            
            datasets.append(dataset)
            print(f"✅ Session {session_id}: {len(dataset)} windows loaded")
            
        except Exception as e:
            print(f"❌ Failed to load session {session_id}: {e}")
            print("   Skipping this session...")
    
    if len(datasets) == 0:
        raise ValueError("No sessions loaded successfully!")
    
    combined_dataset = ConcatDataset(datasets)
    
    print("\n" + "="*70)
    print(f"✅ MULTI-SESSION DATASET CREATED")
    print(f"   Total sessions: {len(datasets)}")
    print(f"   Total windows: {len(combined_dataset)}")
    print(f"   Window shape: (30 neurons, {window_size} timesteps)")
    print("="*70)
    
    return combined_dataset


def create_simple_calcium_dataloaders(batch_size=32, test_split=0.2, num_workers=0, 
                                     window_size=60, stride=50, min_neurons=30, 
                                     use_multi_session=False):
    """
    Factory function per creare dataloaders - supporta multi-sessione.
    
    Args:
        batch_size: batch size
        test_split: frazione per test set
        num_workers: workers per dataloaders
        window_size: dimensione finestre temporali (DEFAULT: 60)
        stride: stride per finestre (DEFAULT: 50)
        min_neurons: numero minimo neuroni (30)
        use_multi_session: se True, usa le 10 sessioni di training
    
    Returns:
        tuple: (train_loader, test_loader, dataset_info)
    """
    from torch.utils.data import DataLoader
    
    print(f"🧠 Creating calcium dataloaders:")
    print(f"   Multi-session: {use_multi_session}")
    print(f"   Parameters: window={window_size}, stride={stride}, neurons={min_neurons}")
    
    if use_multi_session:
        # USA LE 10 NUOVE SESSIONI PER TRAINING
        print(f"   Using {len(TRAINING_SESSION_IDS)} training sessions")
        dataset = create_multi_session_dataset(
            TRAINING_SESSION_IDS,
            window_size=window_size,
            stride=stride,
            min_neurons=min_neurons
        )
    else:
        # Singola sessione (prima delle training sessions)
        print(f"   Using single session: {TRAINING_SESSION_IDS[0]}")
        dataset = SimpleAllenBrainDataset(
            window_size=window_size,
            stride=stride,
            min_neurons=min_neurons,
            session_id=TRAINING_SESSION_IDS[0]
        )
    
    # Split dataset
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, total_size))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Info sulla forma dei dati
    sample = dataset[0]
    neural_shape = sample.shape if isinstance(sample, torch.Tensor) else sample[0].shape
    
    dataset_info = {
        'total_samples': total_size,
        'train_samples': train_size,
        'test_samples': test_size,
        'neural_shape': neural_shape,
        'window_size': window_size,
        'stride': stride,
        'min_neurons': min_neurons,
        'multi_session': use_multi_session,
        'num_sessions': len(TRAINING_SESSION_IDS) if use_multi_session else 1
    }
    
    print(f"✅ Dataloaders created!")
    print(f"   Total samples: {total_size}")
    print(f"   Train: {train_size}, Test: {test_size}")
    print(f"   Neural shape per sample: {neural_shape}")
    
    return train_loader, test_loader, dataset_info


if __name__ == "__main__":
    print("🧠 Testing Multi-Session CalciumDataset...")
    
    # Test multi-session dataset
    print("\n" + "="*70)
    print("TEST 1: Multi-Session Dataset (10 sessions)")
    print("="*70)
    try:
        train_loader, test_loader, info = create_simple_calcium_dataloaders(
            batch_size=16,
            window_size=60,
            stride=50,
            min_neurons=30,
            use_multi_session=True
        )
        
        print(f"\nDataset info: {info}")
        
        # Test batch
        batch = next(iter(train_loader))
        print(f"\nBatch shape: {batch.shape}")
        print("✅ Multi-session dataloader test successful!")
        
    except Exception as e:
        print(f"❌ Multi-session test failed: {e}")
    
    # Test single session
    print("\n" + "="*70)
    print("TEST 2: Single Session Dataset")
    print("="*70)
    try:
        train_loader, test_loader, info = create_simple_calcium_dataloaders(
            batch_size=16,
            window_size=60,
            stride=50,
            min_neurons=30,
            use_multi_session=False
        )
        
        print(f"\nDataset info: {info}")
        
        batch = next(iter(train_loader))
        print(f"Batch shape: {batch.shape}")
        print("✅ Single-session dataloader test successful!")
        
    except Exception as e:
        print(f"❌ Single-session test failed: {e}")