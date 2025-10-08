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
    650389887,  # VISp NUOVA
    642883713,  # VISp NUOVA
    501704220,  # VISp
    501271265,  # VISp
    502205092,  # VISp
    501773889,  # VISp
    502066273,  # VISp
    502115959,  # VISp
    503109347,  # VISp
    502608215,  # VISp
]

# 3 SESSIONI per TEST CROSS-SESSION (quelle già usate prima)
TEST_SESSION_IDS = [
    # 3 VISp (same area as training)
    501559087,  # VISp, Slc17a7-IRES2-Cre
    501498760,  # VISp
    501836392,  # VISp
    
    # 3 VISl (Lateral Visual Area)
    501474098,  # VISl
    663488086,  # Comune
    647593956,  #NO
    
    # 3 VISam - ALTERNATIVE (da testare)
    627823723,  # Comune
    569494121,  # Comune
    566523247,  # Comune
]

# SESSIONE ORIGINALE (ora deprecata, era usata prima per training)
ORIGINAL_TRAINING_SESSION = 501474098

#print(f"📋 Configurate {len(TRAINING_SESSION_IDS)} sessioni per TRAINING")
#print(f"📋 Configurate {len(TEST_SESSION_IDS)} sessioni per TEST CROSS-SESSION")

def select_best_neurons(dff_traces, n_neurons=30, method='variance'):
    """
    Seleziona i migliori n neuroni da un dataset.
    
    Args:
        dff_traces: (total_neurons, time) array
        n_neurons: numero di neuroni da selezionare
        method: 'variance', 'snr', o 'combined'
    
    Returns:
        selected_indices: array degli indici dei neuroni selezionati (ordinati)
        quality_scores: dict con scores di qualità
    """
    total_neurons = dff_traces.shape[0]
    
    if total_neurons <= n_neurons:
        print(f"  ⚠️ Session has only {total_neurons} neurons, using all")
        return np.arange(total_neurons), {}
    
    print(f"  🔍 Selecting best {n_neurons} neurons from {total_neurons} using method='{method}'")
    
    # 1. Calcola varianza (attività neurale)
    variances = np.var(dff_traces, axis=1)
    
    # 2. Calcola SNR (segnale su rumore)
    # SNR = std(signal) / std(noise stimato come differenze temporali)
    snr_scores = np.zeros(total_neurons)
    for i in range(total_neurons):
        trace = dff_traces[i]
        signal_std = np.std(trace)
        noise_std = np.std(np.diff(trace))  # Rumore = differenze temporali
        snr_scores[i] = signal_std / (noise_std + 1e-8)
    
    # 3. Calcola "sparsity" (preferisci neuroni con picchi chiari)
    sparsity_scores = np.zeros(total_neurons)
    for i in range(total_neurons):
        trace = dff_traces[i]
        # Kurtosis misura "peakiness" della distribuzione
        from scipy.stats import kurtosis
        sparsity_scores[i] = kurtosis(trace)
    
    # 4. Combina criteri
    if method == 'variance':
        # Solo varianza
        combined_scores = variances
        
    elif method == 'snr':
        # Solo SNR
        combined_scores = snr_scores
        
    elif method == 'combined':
        # Combina tutti i criteri (normalizzati)
        from sklearn.preprocessing import StandardScaler
        
        # Normalizza ogni metrica
        var_norm = StandardScaler().fit_transform(variances.reshape(-1, 1)).flatten()
        snr_norm = StandardScaler().fit_transform(snr_scores.reshape(-1, 1)).flatten()
        sparse_norm = StandardScaler().fit_transform(sparsity_scores.reshape(-1, 1)).flatten()
        
        # Combina (pesi personalizzabili)
        combined_scores = (
            0.5 * var_norm +      # 50% varianza (attività)
            0.3 * snr_norm +      # 30% SNR (qualità segnale)
            0.2 * sparse_norm     # 20% sparsity (picchi chiari)
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 5. Seleziona top N neuroni
    top_indices = np.argsort(combined_scores)[-n_neurons:]
    
    # Ordina gli indici per mantenere l'ordine originale
    selected_indices = np.sort(top_indices)
    
    # 6. Statistiche di qualità
    quality_scores = {
        'selected_indices': selected_indices.tolist(),
        'method': method,
        'total_neurons': total_neurons,
        'selected_neurons': n_neurons,
        'mean_variance': np.mean(variances[selected_indices]),
        'mean_snr': np.mean(snr_scores[selected_indices]),
        'mean_sparsity': np.mean(sparsity_scores[selected_indices]),
        'variance_percentile': np.percentile(variances, (1 - n_neurons/total_neurons) * 100),
        'excluded_variance': np.mean(variances[~np.isin(np.arange(total_neurons), selected_indices)])
    }
    
    print(f"  ✅ Selected neurons: {selected_indices[:5]}...{selected_indices[-5:]}")
    print(f"     Mean variance (selected): {quality_scores['mean_variance']:.4f}")
    print(f"     Mean variance (excluded): {quality_scores['excluded_variance']:.4f}")
    print(f"     Mean SNR: {quality_scores['mean_snr']:.2f}")
    
    return selected_indices, quality_scores



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
    """Preprocessa solo i dati neurali, SENZA dipendenza dalla velocità"""
    if extraction_result is None:
        print("   ⚠️ No valid data available")
        return None
    
    print("   🔧 Preprocessing neural data...")
    
    dff_data = extraction_result['dff_data']
    dff_timestamps = extraction_result['dff_timestamps']

    # speed_data = extraction_result.get('speed_data')
    # speed_timestamps = extraction_result.get('speed_timestamps')
    # 
    # if speed_data is not None and speed_timestamps is not None and extraction_result['speed_valid']:
    #     speed_aligned = np.interp(dff_timestamps, speed_timestamps, speed_data, 
    #                              left=speed_data[0], right=speed_data[-1])
    #     speed_smooth = gaussian_filter1d(speed_aligned, sigma=5)
    # else:
    #     print("   ⚠️ No valid speed data - using dummy values for neuron selection")
    #     speed_smooth = np.random.randn(len(dff_timestamps)) * 0.1
    
    return {
        'timestamps': dff_timestamps,
        'dff_traces': dff_data,
        # 'speed_for_selection': speed_smooth,
        'metadata': {
            'session_id': extraction_result['session_id'],
            'num_neurons': dff_data.shape[1] if dff_data.ndim == 2 else dff_data.shape[0],
            'num_timepoints': dff_data.shape[0] if dff_data.ndim == 2 else dff_data.shape[1],
            'duration_seconds': dff_timestamps[-1] - dff_timestamps[0],
        }
    }


class SimpleAllenBrainDataset(Dataset):
    """
    Allen Brain Observatory dataset con sliding window su NEURONI e TEMPO.
    Opzione A: Copre TUTTI i neuroni dividendoli in finestre da 30.
    """
    def __init__(self, window_size=60, stride=50, min_neurons=30, 
                 augment=False, session_id=None, 
                 neuron_stride=30,
                 use_neuron_sliding=False,
                 neuron_selection_method='combined'):  
        """
        Args:
            neuron_selection_method: 'variance', 'snr', 'combined', o 'first'
                                    - 'first': primi N neuroni (vecchio comportamento)
                                    - 'variance': seleziona per varianza
                                    - 'snr': seleziona per SNR
                                    - 'combined': combina più criteri (consigliato)
        """
        
        self.window_size = window_size
        self.stride = stride
        self.min_neurons = min_neurons
        self.augment = augment
        self.neuron_stride = neuron_stride
        self.use_neuron_sliding = use_neuron_sliding
        self.neuron_selection_method = neuron_selection_method  # ← NUOVO
        
        if session_id is None:
            self.session_id = TRAINING_SESSION_IDS[0]
        else:
            self.session_id = session_id
        
        # Load and preprocess data
        self._load_allen_data()
    
    def _load_allen_data(self):
        """Load data senza dipendenze dalla velocità."""
        print("🧠 Loading Allen Brain Observatory data...")
        print(f"Using session: {self.session_id}")
        print(f"Parameters: window_size={self.window_size}, stride={self.stride}")
        print(f"🆕 Neuron sliding: {self.use_neuron_sliding}, neuron_stride={self.neuron_stride}")
        
        extraction_result = extract_speed_from_hdf5(self.session_id)
        
        if extraction_result and extraction_result.get('dff_data') is not None:
            processed_data = preprocess_neural_data_only(extraction_result)
            
            if processed_data:
                print("  ✅ Successfully extracted neural data from HDF5!")
                
                timestamps = processed_data['timestamps']
                dff_traces = processed_data['dff_traces']
                
                if dff_traces.ndim == 2 and dff_traces.shape[0] > dff_traces.shape[1]:
                    dff_traces = dff_traces.T
                    print("  🔄 Transposed DFF traces to (neurons, time)")
                
                print(f"  📊 Final neural data shape: {dff_traces.shape}")
                
            else:
                print("  ⚠️ HDF5 preprocessing failed - falling back to Allen SDK")
                timestamps, dff_traces, _ = self._fallback_to_allen_sdk()
        else:
            print("  ⚠️ HDF5 extraction failed - falling back to Allen SDK")
            timestamps, dff_traces, _ = self._fallback_to_allen_sdk()
        
        # 🆕 Preprocessing SENZA padding circolare
        (neural_windows, 
        masks_time, 
        masks_neurons,
        neuron_indices) = self._preprocess_with_neuron_sliding(dff_traces)

        self.neural_data = torch.FloatTensor(neural_windows)
        self.masks_time = torch.BoolTensor(masks_time)
        self.masks_neurons = torch.BoolTensor(masks_neurons)
        self.neuron_indices = neuron_indices
        self.total_neurons = dff_traces.shape[0]

        print(f"📊 Final dataset: {len(self)} windows")
        print(f"   Neural data shape: {self.neural_data.shape}")
        print(f"   Time masks shape: {self.masks_time.shape}")
        print(f"   Neuron masks shape: {self.masks_neurons.shape}")
        print(f"   Total neurons available: {self.total_neurons}")
        print(f"   Complete neuron groups: {len(set([tuple(ni) for ni in neuron_indices]))}")

    def _fallback_to_allen_sdk(self):
        """Fallback per caricare dati usando Allen SDK quando HDF5 fallisce"""
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
        
        print("  📦 Usando Allen SDK come fallback...")
        boc = BrainObservatoryCache()
        
        try:
            data_set = boc.get_ophys_experiment_data(self.session_id)
            timestamps, dff_traces = data_set.get_dff_traces()
            
            print(f"  ✅ Caricati {dff_traces.shape[0]} neuroni, {dff_traces.shape[1]} timepoints via Allen SDK")
            
            try:
                run_ts, running_speed = data_set.get_running_speed()
                
                from scipy.interpolate import interp1d
                speed_interp = interp1d(run_ts, running_speed, 
                                    bounds_error=False, fill_value=0)
                speed_aligned = speed_interp(timestamps)
                
                from scipy.ndimage import gaussian_filter1d
                speed_smooth = gaussian_filter1d(speed_aligned, sigma=5)
                
                print(f"  ✅ Running speed caricato e interpolato")
                
            except Exception as e:
                print(f"  ⚠️ Running speed non disponibile: {e}")
                speed_smooth = np.random.randn(len(timestamps)) * 0.1
            
            return timestamps, dff_traces, speed_smooth
            
        except Exception as e:
            print(f"  ❌ Fallback Allen SDK fallito: {e}")
            raise RuntimeError(f"Impossibile caricare sessione {self.session_id}: {e}")
    
    def _preprocess_with_neuron_sliding(self, dff_traces):
        """
        Preprocessing con selezione intelligente dei neuroni.
        """
        print(f"  🔧 Preprocessing con use_neuron_sliding={self.use_neuron_sliding}")
        print(f"  🔧 Neuron selection method: {self.neuron_selection_method}")
        
        # Normalize neural data
        dff_normalized = zscore(dff_traces, axis=1)
        dff_normalized = np.nan_to_num(dff_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.use_neuron_sliding:
            # 🔵 SLIDING WINDOW SUI NEURONI (per TEST cross-session)
            print(f"  🔵 SLIDING on neurons: stride={self.neuron_stride}, covering all {dff_normalized.shape[0]} neurons")
            (neural_windows, 
            masks_time, 
            masks_neurons,
            neuron_indices) = self._create_neuron_and_temporal_windows(dff_normalized)
        else:
            # 🔴 SELEZIONA MIGLIORI min_neurons NEURONI (per TRAINING)
            print(f"  🔴 SELECTING best {self.min_neurons} neurons from {dff_normalized.shape[0]} total")
            
            if self.neuron_selection_method == 'first':
                # Vecchio comportamento: primi N neuroni
                selected_indices = np.arange(self.min_neurons)
                quality_scores = {'method': 'first'}
                print(f"     Using FIRST {self.min_neurons} neurons (no selection)")
            else:
                # Nuovo: selezione intelligente
                selected_indices, quality_scores = select_best_neurons(
                    dff_normalized, 
                    n_neurons=self.min_neurons,
                    method=self.neuron_selection_method
                )
            
            # Salva info sulla selezione
            self.selected_neuron_indices = selected_indices
            self.neuron_quality_scores = quality_scores
            
            # Estrai solo i neuroni selezionati
            dff_selected = dff_normalized[selected_indices]
            
            # Crea finestre temporali solo su questi neuroni
            neural_windows, masks_time = self._create_neural_windows(dff_selected)
            
            # Tutte le maschere neuroni sono True (tutti neuroni reali)
            masks_neurons = np.ones((len(neural_windows), self.min_neurons), dtype=bool)
            
            # Indici: sempre gli stessi neuroni selezionati
            neuron_indices = [selected_indices.tolist()] * len(neural_windows)
        
        return neural_windows, masks_time, masks_neurons, neuron_indices
    

    def _create_neuron_and_temporal_windows(self, neural_data):
        """
        ✅ CON PADDING CIRCOLARE: Nessuna finestra viene scartata.
        
        Crea finestre temporali E sliding sui neuroni.
        Se mancano neuroni per completare una finestra, usa padding circolare.
        
        Args:
            neural_data: (total_neurons, total_time)
        
        Returns:
            neural_windows: (n_windows, min_neurons, window_size)
            masks_time: (n_windows, window_size) - True=reale, False=padding
            masks_neurons: (n_windows, min_neurons) - True=reale, False=padding
            neuron_indices: list di liste con indici neuroni (include indici circolari)
        """
        total_neurons, total_time = neural_data.shape
        
        neural_windows = []
        masks_time = []
        masks_neurons = []
        neuron_indices = []
        
        # 🔄 Loop sui NEURONI con STRIDE
        neuron_group = 0
        for neuron_start in range(0, total_neurons, self.neuron_stride):
            neuron_end = neuron_start + self.min_neurons
            
            # ✅ NUOVO: Gestisci finestre incomplete con PADDING CIRCOLARE
            if neuron_end > total_neurons:
                # Calcola quanti neuroni mancano
                remaining_neurons = total_neurons - neuron_start
                missing_neurons = self.min_neurons - remaining_neurons
                
                # Dati reali (gli ultimi neuroni disponibili)
                neuron_data_real = neural_data[neuron_start:total_neurons]
                
                # Padding circolare: prendi i primi neuroni per completare
                neuron_data_padding = neural_data[:missing_neurons]
                
                # Concatena: reali + padding
                neuron_data = np.concatenate([neuron_data_real, neuron_data_padding], axis=0)
                
                # Indici dei neuroni (per riferimento/debug)
                indices_real = list(range(neuron_start, total_neurons))
                indices_padding = list(range(missing_neurons))  # Indices circolari
                neuron_indices_group = indices_real + indices_padding
                
                # ✅ Maschera neuroni: True per reali, False per padding
                neuron_mask_base = np.ones(self.min_neurons, dtype=bool)
                neuron_mask_base[remaining_neurons:] = False  # Ultimi neuroni = padding
                
                print(f"  🔄 Neuron group {neuron_group}: [{neuron_start}:{total_neurons}] "
                    f"+ circular padding [0:{missing_neurons}] ({missing_neurons} padded neurons)")
            
            else:
                # ✅ Finestra completa (tutti neuroni reali, nessun padding)
                neuron_indices_group = list(range(neuron_start, neuron_end))
                neuron_data = neural_data[neuron_start:neuron_end]
                
                # Maschera neuroni: tutti True (nessun padding)
                neuron_mask_base = np.ones(self.min_neurons, dtype=bool)
            
            # 🔄 Loop sul TEMPO (sliding temporale per questo gruppo di neuroni)
            temporal_windows, temporal_masks = self._create_temporal_windows_for_neurons(neuron_data)
            
            # Aggiungi tutte le finestre temporali per questo gruppo di neuroni
            for tw, tm in zip(temporal_windows, temporal_masks):
                neural_windows.append(tw)
                masks_time.append(tm)
                masks_neurons.append(neuron_mask_base)  # ← Stessa maschera per tutte le finestre temporali
                neuron_indices.append(neuron_indices_group)
            
            neuron_group += 1
        
        print(f"  ✅ Created {neuron_group} neuron groups (including padded) "
            f"× ~{len(temporal_windows)} temporal windows = {len(neural_windows)} total windows")
        
        return (np.array(neural_windows), 
                np.array(masks_time), 
                np.array(masks_neurons),
                neuron_indices)
    
    
    def _create_temporal_windows_for_neurons(self, neuron_data):
        """
        Crea sliding temporal windows per un gruppo di neuroni.
        
        Args:
            neuron_data: (min_neurons, total_time)
        
        Returns:
            temporal_windows: list di (min_neurons, window_size)
            temporal_masks: list di (window_size,)
        """
        temporal_windows = []
        temporal_masks = []
        
        _, total_time = neuron_data.shape
        
        # Calcola finestre complete
        num_complete_windows = (total_time - self.window_size) // self.stride + 1
        last_complete_start = (num_complete_windows - 1) * self.stride
        
        # Crea finestre complete
        for start in range(0, total_time - self.window_size + 1, self.stride):
            window = neuron_data[:, start:start + self.window_size]
            mask = np.ones(self.window_size, dtype=bool)
            
            temporal_windows.append(window)
            temporal_masks.append(mask)
        
        # 🔄 Ultima finestra con padding se necessario
        next_start = last_complete_start + self.stride
        if next_start < total_time:
            remaining_data = neuron_data[:, next_start:]
            missing_timesteps = self.window_size - remaining_data.shape[1]
            
            # Padding circolare
            padding = neuron_data[:, :missing_timesteps]
            padded_window = np.concatenate([remaining_data, padding], axis=1)
            
            # Maschera
            mask = np.ones(self.window_size, dtype=bool)
            mask[-missing_timesteps:] = False
            
            temporal_windows.append(padded_window)
            temporal_masks.append(mask)
        
        return temporal_windows, temporal_masks
    
    def _create_neural_windows(self, neural_data):
        """
        🔴 VECCHIO: Solo sliding temporale (per retrocompatibilità).
        """
        neural_windows = []
        masks = []
        total_time = neural_data.shape[1]
        
        num_complete_windows = (total_time - self.window_size) // self.stride + 1
        last_complete_start = (num_complete_windows - 1) * self.stride
        
        for start in range(0, total_time - self.window_size + 1, self.stride):
            neural_window = neural_data[:, start:start + self.window_size]
            neural_windows.append(neural_window)
            
            mask = np.ones(self.window_size, dtype=bool)
            masks.append(mask)
        
        next_start = last_complete_start + self.stride
        if next_start < total_time:
            remaining_data = neural_data[:, next_start:]
            missing_timesteps = self.window_size - remaining_data.shape[1]
            
            padding = neural_data[:, :missing_timesteps]
            padded_window = np.concatenate([remaining_data, padding], axis=1)
            neural_windows.append(padded_window)
            
            mask = np.ones(self.window_size, dtype=bool)
            mask[-missing_timesteps:] = False
            masks.append(mask)
        
        return np.array(neural_windows), np.array(masks)
    
    def __len__(self):
        return len(self.neural_data)
    
    def __getitem__(self, idx):
        """
        AGGIORNATO: Restituisce finestra con ENTRAMBE le maschere.
        """
        neural = self.neural_data[idx]  # Shape: (min_neurons, window_size)
        mask_time = self.masks_time[idx]  # Shape: (window_size,)
        mask_neurons = self.masks_neurons[idx]  # 🆕 Shape: (min_neurons,)
        
        # Applica augmentation se richiesta
        if self.augment:
            neural = self._augment_neural_data(neural)
        
        return neural, mask_time, mask_neurons  # 🆕 Restituisci entrambe le maschere
    
    def get_neuron_indices(self, idx):
        """
        🆕 NUOVO: Restituisce gli indici dei neuroni per questa finestra.
        """
        return self.neuron_indices[idx]
        
    def _augment_neural_data(self, neural_data):
        """Apply minimal augmentation specific to Allen Brain data."""
        if np.random.rand() < 0.3:
            noise_std = 0.03 * torch.std(neural_data)
            noise = torch.randn_like(neural_data) * noise_std
            neural_data = neural_data + noise
        
        return neural_data


def create_multi_session_dataset(session_ids, window_size=60, stride=50, min_neurons=30,
                                use_neuron_sliding=False,
                                neuron_selection_method='combined'):  # ← NUOVO
    """
    Args:
        neuron_selection_method: come selezionare neuroni quando use_neuron_sliding=False
                                'combined' (default): usa criteri multipli
                                'variance': solo varianza
                                'snr': solo signal-to-noise
                                'first': primi N neuroni (non consigliato)
    """
    print(f"\n🎯 Creating MULTI-SESSION dataset from {len(session_ids)} sessions")
    print(f"   Window size: {window_size}, Stride: {stride}, Min neurons: {min_neurons}")
    print(f"   🎲 Neuron sliding: {use_neuron_sliding}")
    if not use_neuron_sliding:
        print(f"   🔍 Neuron selection: {neuron_selection_method}")
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
                session_id=session_id,
                use_neuron_sliding=use_neuron_sliding,
                neuron_selection_method=neuron_selection_method  # ← PASSA IL PARAMETRO
            )
            
            datasets.append(dataset)
            print(f"✅ Session {session_id}: {len(dataset)} windows loaded")
            
            # Stampa info sulla selezione neuroni
            if not use_neuron_sliding and hasattr(dataset, 'neuron_quality_scores'):
                scores = dataset.neuron_quality_scores
                print(f"   📊 Neuron quality: variance={scores.get('mean_variance', 0):.4f}, "
                      f"SNR={scores.get('mean_snr', 0):.2f}")
            
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
    print(f"   Window shape: ({min_neurons} neurons, {window_size} timesteps)")
    # ✅ NUOVO
    if use_neuron_sliding:
        if hasattr(dataset, 'total_neurons'):
            print(f"   🔵 SLIDING: Using neuron groups from {dataset.total_neurons} total neurons")
    else:
        print(f"   🔴 SELECTION: Using best {min_neurons} neurons per session ({neuron_selection_method})")
    
    return combined_dataset




def create_simple_calcium_dataloaders(batch_size=32, test_split=0.2, num_workers=0, 
                                     window_size=60, stride=50, min_neurons=30, 
                                     use_multi_session=False,
                                     use_neuron_sliding=False,
                                     neuron_selection_method='combined'):  # ← NUOVO
    """
    Factory function per creare dataloaders.
    
    Args:
        neuron_selection_method: 'combined' (default), 'variance', 'snr', o 'first'
    """
    from torch.utils.data import DataLoader
    
    print(f"🧠 Creating calcium dataloaders:")
    print(f"   Multi-session: {use_multi_session}")
    print(f"   Neuron sliding: {use_neuron_sliding}")
    if not use_neuron_sliding:
        print(f"   Neuron selection: {neuron_selection_method}")
    print(f"   Parameters: window={window_size}, stride={stride}, neurons={min_neurons}")
    
    if use_multi_session:
        print(f"   Using {len(TRAINING_SESSION_IDS)} training sessions")
        dataset = create_multi_session_dataset(
            TRAINING_SESSION_IDS,
            window_size=window_size,
            stride=stride,
            min_neurons=min_neurons,
            use_neuron_sliding=use_neuron_sliding,
            neuron_selection_method=neuron_selection_method  # ← PASSA IL PARAMETRO
        )
    else:
        print(f"   Using single session: {TRAINING_SESSION_IDS[0]}")
        dataset = SimpleAllenBrainDataset(
            window_size=window_size,
            stride=stride,
            min_neurons=min_neurons,
            session_id=TRAINING_SESSION_IDS[0],
            use_neuron_sliding=use_neuron_sliding,
            neuron_selection_method=neuron_selection_method  # ← PASSA IL PARAMETRO
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
    
    # Info
    sample = dataset[0]
    neural_shape = sample[0].shape if isinstance(sample, (list, tuple)) else sample.shape
    
    dataset_info = {
    'total_samples': total_size,
    'train_samples': train_size,
    'test_samples': test_size,
    'neural_shape': neural_shape,
    'window_size': window_size,
    'stride': stride,
    'min_neurons': min_neurons,
    'multi_session': use_multi_session,
    'num_sessions': len(TRAINING_SESSION_IDS) if use_multi_session else 1,
    'use_neuron_sliding': use_neuron_sliding,
    'neuron_selection_method': neuron_selection_method,  # ← AGGIUNGI QUESTA RIGA
    'total_neurons_available': dataset.total_neurons if hasattr(dataset, 'total_neurons') else None
    }
    
    print(f"✅ Dataloaders created!")
    print(f"   Total samples: {total_size}")
    print(f"   Train: {train_size}, Test: {test_size}")
    print(f"   Neural shape per sample: {neural_shape}")
    if use_neuron_sliding:
        if hasattr(dataset, 'total_neurons'):
            print(f"   🔵 SLIDING: Using neuron groups from {dataset.total_neurons} total neurons")
    else:
        print(f"   🔴 FIXED: Using only first {min_neurons} neurons per session")
    
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