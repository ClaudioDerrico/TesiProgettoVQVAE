import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, zscore
from sklearn.preprocessing import StandardScaler, RobustScaler
import h5py
import gc
import warnings
warnings.filterwarnings('ignore')


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
            
        return neural  # ✅ SOLO neural data, no behavior
    
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
    """
    Extract running speed directly from HDF5/NWB file.
    ✅ MANTENUTO per compatibilità con Allen Brain Observatory
    """
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    
    print(f"🔍 Extracting data directly from HDF5 for session {session_id}...")
    
    boc = BrainObservatoryCache()
    
    try:
        # Get dataset object to access NWB file path
        ds = boc.get_ophys_experiment_data(session_id)
        fpath = ds.nwb_file
        print(f"   📁 NWB file path: {fpath}")
        
        # Clean up dataset object immediately
        del ds
        gc.collect()
        
        # Open HDF5 file directly with cache disabled
        with h5py.File(fpath, 'r', rdcc_nbytes=0) as f:
            
            # HDF5 paths
            B = 'processing/brain_observatory_pipeline'
            speed_path = f'{B}/BehavioralTimeSeries/running_speed/data'
            speed_ts_path = f'{B}/BehavioralTimeSeries/running_speed/timestamps'
            dff_path = f'{B}/DfOverF/imaging_plane_1/data'
            dff_ts_path = f'{B}/DfOverF/imaging_plane_1/timestamps'
            
            # Extract running speed data (for neuron selection)
            if speed_path in f and speed_ts_path in f:
                print("   ✅ Found running speed data in HDF5")
                
                # Read speed data and timestamps
                speed_data = f[speed_path][:]
                speed_timestamps = f[speed_ts_path][:]
                
                print(f"   📊 Speed data shape: {speed_data.shape}")
                print(f"   📈 Speed range: {np.nanmin(speed_data):.3f} - {np.nanmax(speed_data):.3f}")
                
                # Check for NaN values
                nan_count = np.sum(np.isnan(speed_data))
                valid_pct = 100 * (len(speed_data) - nan_count) / len(speed_data)
                print(f"   ✅ Valid speed data: {valid_pct:.1f}% ({len(speed_data) - nan_count}/{len(speed_data)})")
                
                # Check if speed data is realistic
                speed_range = np.nanmax(speed_data) - np.nanmin(speed_data)
                realistic_speed = (0.1 < speed_range < 100)
                speed_valid = realistic_speed and valid_pct > 50
                
            else:
                print("   ❌ Running speed data not found in HDF5")
                speed_data, speed_timestamps = None, None
                speed_valid = False
            
            # Extract neural data
            if dff_path in f and dff_ts_path in f:
                print("   ✅ Found neural data in HDF5")
                
                dff_data = f[dff_path][:]
                dff_timestamps = f[dff_ts_path][:]
                
                # Handle potential transpose issue
                if dff_data.ndim == 2 and dff_data.shape[0] != dff_timestamps.shape[0]:
                    dff_data = dff_data.T
                    print("   🔄 Transposed DFF data for correct alignment")
                
                print(f"   📊 DFF data shape: {dff_data.shape}")
                
            else:
                print("   ❌ Neural data not found in HDF5")
                dff_data, dff_timestamps = None, None
        
        # Return extracted data
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
    """
    ✅ SEMPLIFICATO: Preprocessa solo i dati neurali, speed solo per selezione neuroni
    """
    if extraction_result is None:
        print("   ⚠️ No valid data available")
        return None
    
    print("   🔧 Preprocessing neural data...")
    
    dff_data = extraction_result['dff_data']
    dff_timestamps = extraction_result['dff_timestamps']
    speed_data = extraction_result.get('speed_data')
    speed_timestamps = extraction_result.get('speed_timestamps')
    
    # Align speed data solo per selezione neuroni (se disponibile)
    if speed_data is not None and speed_timestamps is not None and extraction_result['speed_valid']:
        speed_aligned = np.interp(dff_timestamps, speed_timestamps, speed_data, 
                                 left=speed_data[0], right=speed_data[-1])
        speed_smooth = gaussian_filter1d(speed_aligned, sigma=5)
    else:
        print("   ⚠️ No valid speed data - using dummy values for neuron selection")
        speed_smooth = np.random.randn(len(dff_timestamps)) * 0.1  # Dummy speed data
    
    return {
        'timestamps': dff_timestamps,
        'dff_traces': dff_data,
        'speed_for_selection': speed_smooth,  # Solo per selezione, non output
        'metadata': {
            'session_id': extraction_result['session_id'],
            'num_neurons': dff_data.shape[1] if dff_data.ndim == 2 else dff_data.shape[0],
            'num_timepoints': dff_data.shape[0] if dff_data.ndim == 2 else dff_data.shape[1],
            'duration_seconds': dff_timestamps[-1] - dff_timestamps[0],
        }
    }


class SimpleAllenBrainDataset(Dataset):
    """
    ✅ SIMPLIFIED: Allen Brain Observatory dataset focused only on neural data reconstruction.
    """
    
    def __init__(self, window_size=50, stride=10, min_neurons=30, augment=False):
        """
        Args:
            window_size: size of temporal windows
            stride: stride for sliding windows
            min_neurons: minimum number of neurons to keep
            augment: whether to apply minimal augmentation
        """
        self.session_id = 501474098  # VISp session
        self.window_size = window_size
        self.stride = stride
        self.min_neurons = min_neurons
        self.augment = augment
        
        # Load and preprocess data
        self._load_allen_data()
        
    def _load_allen_data(self):
        """✅ SIMPLIFIED: Load data focusing only on neural reconstruction."""
        print("🧠 Loading Allen Brain Observatory data for reconstruction...")
        print(f"Using session: {self.session_id}")
        
        # Extract data directly from HDF5
        extraction_result = extract_speed_from_hdf5(self.session_id)
        
        if extraction_result:
            # Process extracted data
            processed_data = preprocess_neural_data_only(extraction_result)
            
            if processed_data:
                print("  ✅ Successfully extracted neural data from HDF5!")
                
                timestamps = processed_data['timestamps']
                dff_traces = processed_data['dff_traces']
                speed_for_selection = processed_data['speed_for_selection']
                
                # Ensure correct shape (neurons, time)
                if dff_traces.ndim == 2 and dff_traces.shape[0] > dff_traces.shape[1]:
                    dff_traces = dff_traces.T
                    print("  🔄 Transposed DFF traces to (neurons, time)")
                
                print(f"  📊 Final neural data shape: {dff_traces.shape}")
                
            else:
                # Fallback to standard Allen SDK
                print("  ⚠️ HDF5 preprocessing failed - falling back to Allen SDK")
                timestamps, dff_traces, speed_for_selection = self._fallback_to_allen_sdk()
        else:
            # Fallback to standard Allen SDK
            print("  ⚠️ HDF5 extraction failed - falling back to Allen SDK")
            timestamps, dff_traces, speed_for_selection = self._fallback_to_allen_sdk()
        
        # ✅ SOLO preprocessing neural data
        neural_windows = self._preprocess_neural_only(dff_traces, speed_for_selection)
        
        # Convert to tensors
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
        
        # Align speed for neuron selection
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
        
        # Select active neurons (using speed for correlation)
        active_neurons = self._select_active_neurons(dff_traces, speed_for_selection)
        dff_active = dff_traces[active_neurons, :]
        
        print(f"  ✅ Selected {np.sum(active_neurons)} active neurons")
        
        # Normalize neural data
        dff_normalized = zscore(dff_active, axis=1)
        dff_normalized = np.nan_to_num(dff_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create temporal windows - ✅ SOLO NEURAL DATA
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
            combined_score[i] = 0.7 * var_percentile + 0.3 * corr_percentile  # Più peso alla varianza
        
        # Select top neurons
        top_neurons = np.argsort(combined_score)[-self.min_neurons:]
        active_neurons = np.zeros(len(dff_traces), dtype=bool)
        active_neurons[top_neurons] = True
        
        print(f"  📊 Neuron selection stats:")
        print(f"     Mean correlation: {np.mean(correlations[active_neurons]):.3f}")
        print(f"     Mean variance: {np.mean(variances[active_neurons]):.3f}")
        
        return active_neurons
    
    def _create_neural_windows(self, neural_data):
        
        neural_windows = []
        
        for start in range(0, neural_data.shape[1] - self.window_size + 1, self.stride):
            # Solo neural window
            neural_window = neural_data[:, start:start + self.window_size]
            neural_windows.append(neural_window)
        
        return np.array(neural_windows)
    
    def __len__(self):
        return len(self.neural_data)
    
    def __getitem__(self, idx):
        neural = self.neural_data[idx]
        
        # Apply minimal augmentation if enabled
        if self.augment:
            neural = self._augment_neural_data(neural)
        
        return neural  # ✅ SOLO neural data
    
    def _augment_neural_data(self, neural_data):
        """Apply minimal augmentation specific to Allen Brain data."""
        if np.random.rand() < 0.3:
            # Add realistic neural noise
            noise_std = 0.03 * torch.std(neural_data)
            noise = torch.randn_like(neural_data) * noise_std
            neural_data = neural_data + noise
        
        return neural_data


def create_simple_calcium_dataloaders(batch_size=32, test_split=0.3, num_workers=0, 
                                     window_size=50, stride=10, min_neurons=30, augment_train=False):
    """
    ✅ SIMPLIFIED: Factory function for calcium dataloaders - ONLY neural data.
    
    Args:
        batch_size: batch size for dataloaders
        test_split: fraction for test set
        num_workers: number of workers for dataloaders
        window_size: size of temporal windows
        stride: stride for sliding windows
        min_neurons: minimum number of neurons to keep
        augment_train: apply augmentation to training data
    
    Returns:
        tuple: (train_loader, test_loader, dataset_info)
    """
    from torch.utils.data import DataLoader
    
    print(f"🧠 Creating SIMPLIFIED calcium dataloaders:")
    print(f"   Session: VISp (501474098)")
    print(f"   Split: {int((1-test_split)*100)}% train, {int(test_split*100)}% test")
    print(f"   Window size: {window_size}, Min neurons: {min_neurons}")
    
    # Create dataset
    dataset = SimpleAllenBrainDataset(
        window_size=window_size,
        stride=stride,
        min_neurons=min_neurons,
        augment=False  # No augmentation during dataset creation
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
    
    dataset_info = {
        'total_samples': total_size,
        'train_samples': train_size,
        'test_samples': test_size,
        'neural_shape': dataset.neural_data.shape[1:],  # (neurons, timesteps)
        'session_id': dataset.session_id,
        'window_size': window_size,
        'min_neurons': min_neurons
    }
    
    print(f"✅ Dataloaders created successfully!")
    print(f"   Total samples: {total_size}")
    print(f"   Neural shape per sample: {dataset_info['neural_shape']}")
    
    return train_loader, test_loader, dataset_info


if __name__ == "__main__":
    print("🧠 Testing CalciumDataset...")
    
    # Test basic dataset
    neural_data = np.random.randn(100, 30, 50)  # 100 samples, 30 neurons, 50 timepoints
    dataset = CalciumDataset(neural_data, augment=True)
    print(f"Basic dataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    
    # Test Allen Brain dataset
    print("\n🔬 Testing SimpleAllenBrainDataset...")
    try:
        allen_dataset = SimpleAllenBrainDataset(window_size=50, stride=25, min_neurons=20)
        print(f"Allen dataset length: {len(allen_dataset)}")
        
        sample = allen_dataset[0]
        print(f"Allen sample shape: {sample.shape}")
    except Exception as e:
        print(f"Allen dataset test failed: {e}")
    
    # Test dataloader creation
    print("\n📦 Testing dataloader creation...")
    try:
        train_loader, test_loader, info = create_simple_calcium_dataloaders(
            batch_size=16,
            window_size=50,
            min_neurons=20
        )
        
        print(f"Dataset info: {info}")
        
        # Test batch
        batch = next(iter(train_loader))
        print(f"Batch shape: {batch.shape}")
        print("✅ Dataloader creation successful!")
        
    except Exception as e:
        print(f"Dataloader test failed: {e}")