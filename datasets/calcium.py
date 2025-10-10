import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import h5py
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE SESSIONI
# ============================================================================

# 10 SESSIONI per TRAINING
TRAINING_SESSION_IDS = [
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

# 3 SESSIONI per TEST CROSS-SESSION
TEST_SESSION_IDS = [
    501559087,  # VISp
    501498760,  # VISp
    501836392,  # VISp
]


def select_random_neurons(dff_traces, n_neurons=30, seed=42):
    """
    Seleziona N neuroni casuali.
    
    Args:
        dff_traces: (total_neurons, time) array
        n_neurons: numero di neuroni da selezionare
        seed: seed per riproducibilità (None = casuale)
    
    Returns:
        selected_indices: array degli indici dei neuroni selezionati
    """
    total_neurons = dff_traces.shape[0]
    
    if total_neurons <= n_neurons:
        print(f"  ⚠️ Session has only {total_neurons} neurons, using all")
        return np.arange(total_neurons)
    
    print(f"  🎲 Selecting {n_neurons} random neurons from {total_neurons}")
    
    if seed is not None:
        np.random.seed(seed)
    
    selected_indices = np.random.choice(total_neurons, n_neurons, replace=False)
    selected_indices = np.sort(selected_indices)
    
    print(f"  ✅ Selected: {selected_indices[:5].tolist()}...{selected_indices[-5:].tolist()}")
    
    return selected_indices


def extract_neural_data_from_hdf5(session_id):
    """Estrae SOLO dati neurali dall'HDF5."""
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    
    print(f"🔍 Extracting neural data for session {session_id}...")
    
    boc = BrainObservatoryCache()
    
    try:
        ds = boc.get_ophys_experiment_data(session_id)
        fpath = ds.nwb_file
        print(f"   📁 NWB file: {fpath}")
        
        del ds
        gc.collect()
        
        with h5py.File(fpath, 'r', rdcc_nbytes=0) as f:
            B = 'processing/brain_observatory_pipeline'
            dff_path = f'{B}/DfOverF/imaging_plane_1/data'
            dff_ts_path = f'{B}/DfOverF/imaging_plane_1/timestamps'
            
            if dff_path in f and dff_ts_path in f:
                print("   ✅ Found neural data")
                
                dff_data = f[dff_path][:]
                dff_timestamps = f[dff_ts_path][:]

                
                print(f"   📊 Shape: {dff_data.shape}")
                
                return {
                    'dff_data': dff_data,
                    'timestamps': dff_timestamps,
                    'session_id': session_id,
                    'success': True
                }
            else:
                print("   ❌ Neural data not found")
                return None
                
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        return None


class SimpleAllenBrainDataset(Dataset):
    """
    Dataset minimale per Allen Brain Observatory.
    Carica dati neurali e crea sliding windows temporali.
    """
    
    def __init__(self, window_size=60, stride=50, min_neurons=30, 
                 session_id=None, neuron_stride=30, use_neuron_sliding=False):
        
        self.window_size = window_size
        self.stride = stride
        self.min_neurons = min_neurons
        self.neuron_stride = neuron_stride
        self.use_neuron_sliding = use_neuron_sliding
        
        if session_id is None:
            self.session_id = TRAINING_SESSION_IDS[0]
        else:
            self.session_id = session_id
        
        self._load_allen_data()
    
    def _load_allen_data(self):
        """Carica dati neurali e crea finestre COMPLETE."""
        print(f"🧠 Loading session: {self.session_id}")
        print(f"   Parameters: window={self.window_size}, stride={self.stride}")
        print(f"   Mode: {'SLIDING on neurons (TEST)' if self.use_neuron_sliding else 'RANDOM 30 neurons (TRAIN)'}")
        
        # Estrai dati grezzi
        extraction_result = extract_neural_data_from_hdf5(self.session_id)
        
        if extraction_result is None or not extraction_result['success']:
            raise RuntimeError(f"Failed to load session {self.session_id}")
        
        dff_traces = extraction_result['dff_data']
        print(f"  📊 Raw data from HDF5: {dff_traces.shape}")

        # ✅ LOGICA CORRETTA: neuroni < timepoints nel calcium imaging
        n1, n2 = dff_traces.shape
        
        # Euristica: la dimensione PIÙ GRANDE è quasi sempre TIME
        if n1 > n2:
            # n1 è probabilmente TIME → trasponi
            print(f"  🔄 Detected (time={n1}, neurons={n2}) → Transposing...")
            dff_traces = dff_traces.T
            n_neurons = n2
            n_timepoints = n1
        else:
            # n1 è probabilmente NEURONS → già corretto
            print(f"  ✅ Detected (neurons={n1}, time={n2}) → No transpose needed")
            n_neurons = n1
            n_timepoints = n2
        
        print(f"  ✅ Final shape: ({n_neurons} neurons, {n_timepoints} timepoints)")
        print(f"  📊 Recording duration: ~{n_timepoints/30:.1f}s @ 30Hz")
        
        # Sanity checks
        if n_neurons > 1000:
            print(f"  ⚠️ WARNING: Unusually high neuron count ({n_neurons})")
        if n_timepoints < 1000:
            print(f"  ⚠️ WARNING: Very short recording ({n_timepoints} timepoints)")
        
        # Ora dff_traces è GARANTITO essere (neurons, time)
        
        if self.use_neuron_sliding:
            # TEST MODE
            neural_windows, masks_time, masks_neurons, neuron_indices = \
                self._create_neuron_and_temporal_windows(dff_traces)
            
            self.neural_data = torch.FloatTensor(neural_windows)
            self.masks_time = torch.BoolTensor(masks_time)
            self.masks_neurons = torch.BoolTensor(masks_neurons)
            self.neuron_indices = neuron_indices
            
        else:
            # TRAINING MODE
            selected_indices = select_random_neurons(dff_traces, self.min_neurons)
            dff_selected = dff_traces[selected_indices]  # (30, time)
            
            neural_windows = self._create_temporal_windows(dff_selected)
            
            self.neural_data = torch.FloatTensor(neural_windows)
            self.masks_time = None
            self.masks_neurons = None
            self.neuron_indices = [selected_indices.tolist()] * len(neural_windows)
        
        self.total_neurons = n_neurons  # ← Usa la variabile corretta!
        
        print(f"✅ Dataset ready: {len(self.neural_data)} windows")
        print(f"   Shape: {self.neural_data.shape}")


    def _create_temporal_windows(self, neural_data):
        """
        Crea sliding temporal windows SOLO COMPLETE (training mode).
        SCARTA l'ultima finestra se incompleta.
        
        Args:
            neural_data: (neurons, time)
        
        Returns:
            neural_windows: (n_windows, neurons, window_size)
        """
        neural_windows = []
        total_time = neural_data.shape[1]
        
        # ✅ SOLO finestre COMPLETE - no padding!
        for start in range(0, total_time - self.window_size + 1, self.stride):
            window = neural_data[:, start:start + self.window_size]
            
            # Verifica che sia completa
            if window.shape[1] == self.window_size:
                neural_windows.append(window)
        
        print(f"  📊 Created {len(neural_windows)} COMPLETE windows from {total_time} timepoints")
        
        return np.array(neural_windows)
    

    def _create_neuron_and_temporal_windows(self, neural_data):
        """
        Crea sliding windows su NEURONI e TEMPO (per test cross-session).
        Con padding circolare sui neuroni.
        
        Args:
            neural_data: (total_neurons, total_time)
        
        Returns:
            neural_windows, masks_time, masks_neurons, neuron_indices
        """
        total_neurons, total_time = neural_data.shape
        
        neural_windows = []
        masks_time = []
        masks_neurons = []
        neuron_indices = []
        
        # Loop sui NEURONI
        for neuron_start in range(0, total_neurons, self.neuron_stride):
            neuron_end = neuron_start + self.min_neurons
            
            # Padding circolare se necessario
            if neuron_end > total_neurons:
                remaining = total_neurons - neuron_start
                missing = self.min_neurons - remaining
                
                # Dati reali + padding circolare
                neuron_data = np.concatenate([
                    neural_data[neuron_start:total_neurons],
                    neural_data[:missing]
                ], axis=0)
                
                # Maschera neuroni
                neuron_mask = np.ones(self.min_neurons, dtype=bool)
                neuron_mask[remaining:] = False
                
                # Indici
                indices = list(range(neuron_start, total_neurons)) + list(range(missing))
            else:
                # Finestra completa
                neuron_data = neural_data[neuron_start:neuron_end]
                neuron_mask = np.ones(self.min_neurons, dtype=bool)
                indices = list(range(neuron_start, neuron_end))
            
            # Loop sul TEMPO per questo gruppo di neuroni
            temporal_windows, temporal_masks = self._create_temporal_windows(neuron_data)
            
            for tw, tm in zip(temporal_windows, temporal_masks):
                neural_windows.append(tw)
                masks_time.append(tm)
                masks_neurons.append(neuron_mask)
                neuron_indices.append(indices)
        
        return (np.array(neural_windows), 
                np.array(masks_time), 
                np.array(masks_neurons),
                neuron_indices)
    
    def __len__(self):
        return len(self.neural_data)
    
    def __getitem__(self, idx):
        """
        Returns:
        - TRAINING mode: solo neural_data (Tensor)
        - TEST mode: (neural_data, mask_time, mask_neurons)
        """
        if self.use_neuron_sliding:
            # TEST: con maschere
            return (self.neural_data[idx], 
                    self.masks_time[idx], 
                    self.masks_neurons[idx])
        else:
            # TRAINING: solo dati (no maschere)
            return self.neural_data[idx]


def create_multi_session_dataset(session_ids, window_size=60, stride=50, 
                                min_neurons=30, use_neuron_sliding=False):
    """
    Crea dataset da multiple sessioni.
    
    Args:
        session_ids: lista di session IDs
        window_size: dimensione finestra temporale
        stride: stride per sliding window
        min_neurons: numero neuroni per finestra
        use_neuron_sliding: True per TEST (sliding), False per TRAINING (random)
    
    Returns:
        ConcatDataset
    """
    print(f"\n🎯 Creating MULTI-SESSION dataset from {len(session_ids)} sessions")
    print(f"   Window: {window_size}, Stride: {stride}, Neurons: {min_neurons}")
    print(f"   Mode: {'SLIDING neurons (TEST)' if use_neuron_sliding else 'RANDOM neurons (TRAIN)'}")
    print("="*70)
    
    datasets = []
    
    for i, session_id in enumerate(session_ids):
        print(f"\n📊 Session {i+1}/{len(session_ids)}: {session_id}")
        print("-"*70)
        
        try:
            dataset = SimpleAllenBrainDataset(
                window_size=window_size,
                stride=stride,
                min_neurons=min_neurons,
                session_id=session_id,
                use_neuron_sliding=use_neuron_sliding
            )
            
            datasets.append(dataset)
            print(f"✅ Loaded: {len(dataset)} windows")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    if len(datasets) == 0:
        raise ValueError("No sessions loaded!")
    
    combined = ConcatDataset(datasets)
    
    print("\n" + "="*70)
    print(f"✅ MULTI-SESSION DATASET CREATED")
    print(f"   Sessions: {len(datasets)}")
    print(f"   Total windows: {len(combined)}")
    print(f"   Shape per window: ({min_neurons} neurons, {window_size} timesteps)")
    
    return combined


def create_simple_calcium_dataloaders(batch_size=64, test_split=0.2, num_workers=0, 
                                     window_size=60, stride=50, min_neurons=30, 
                                     use_multi_session=False, use_neuron_sliding=False):
    """
    Factory function per creare dataloaders.
    
    Args:
        batch_size: batch size
        test_split: frazione per test set
        num_workers: workers per dataloader
        window_size: finestra temporale
        stride: stride temporale
        min_neurons: numero neuroni
        use_multi_session: True per multi-sessione
        use_neuron_sliding: True per sliding (TEST), False per random (TRAIN)
    
    Returns:
        train_loader, test_loader, dataset_info
    """
    from torch.utils.data import DataLoader
    
    print(f"🧠 Creating calcium dataloaders:")
    print(f"   Multi-session: {use_multi_session}")
    print(f"   Neuron mode: {'SLIDING (test)' if use_neuron_sliding else 'RANDOM (train)'}")
    
    if use_multi_session:
        print(f"   Using {len(TRAINING_SESSION_IDS)} training sessions")
        dataset = create_multi_session_dataset(
            TRAINING_SESSION_IDS,
            window_size=window_size,
            stride=stride,
            min_neurons=min_neurons,
            use_neuron_sliding=use_neuron_sliding
        )
    else:
        print(f"   Using single session: {TRAINING_SESSION_IDS[0]}")
        dataset = SimpleAllenBrainDataset(
            window_size=window_size,
            stride=stride,
            min_neurons=min_neurons,
            session_id=TRAINING_SESSION_IDS[0],
            use_neuron_sliding=use_neuron_sliding
        )
    
    # Split
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, total_size))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Info
    dataset_info = {
        'total_samples': total_size,
        'train_samples': train_size,
        'test_samples': test_size,
        'neural_shape': (min_neurons, window_size),
        'window_size': window_size,
        'stride': stride,
        'min_neurons': min_neurons,
        'multi_session': use_multi_session,
        'num_sessions': len(TRAINING_SESSION_IDS) if use_multi_session else 1,
        'use_neuron_sliding': use_neuron_sliding,
    }
    
    print(f"\n✅ Dataloaders created!")
    print(f"   Train: {train_size}, Test: {test_size}")
    
    return train_loader, test_loader, dataset_info