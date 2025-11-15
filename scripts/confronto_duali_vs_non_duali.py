#!/usr/bin/env python3
"""
Plot codebook size vs metrics (correlation and MSE)
Comparing DUAL vs NON-DUAL models
Aggregated by session type: VISP, VISL, VISAM
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
OUTPUT_DIR = Path(r"C:\Users\Piedi\Desktop\TesiProgettoVQVAE\confronto_duali_vs_non_duali")

# Mapping session types based on position in results
# First 3 sessions: VISP
# Next 3 sessions: VISL  
# Last 3 sessions: VISAM
SESSION_TYPE_INDICES = {
    'VISP': [0, 1, 2],
    'VISL': [3, 4, 5],
    'VISAM': [6, 7, 8]
}

# ============================================================================
# DATI NON-DUALI (Single Neuron)
# ============================================================================

METRICS_DATA_NON_DUAL = {
    16: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.003831, "corr_mean": 0.7683, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.001166, "corr_mean": 0.7020, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000930, "corr_mean": 0.7482, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.001694, "corr_mean": 0.7387, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000858, "corr_mean": 0.7288, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.002090, "corr_mean": 0.6937, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.001445, "corr_mean": 0.6709, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.003935, "corr_mean": 0.6826, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.001418, "corr_mean": 0.6654, "samples": 2486}
        ]
    },
    32: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.003056, "corr_mean": 0.7911, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.001042, "corr_mean": 0.7292, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000791, "corr_mean": 0.7763, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.001340, "corr_mean": 0.7648, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000731, "corr_mean": 0.7577, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.001911, "corr_mean": 0.7213, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.001319, "corr_mean": 0.7014, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.003571, "corr_mean": 0.7106, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.001298, "corr_mean": 0.6959, "samples": 2486}
        ]
    },
    64: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.003078, "corr_mean": 0.8928, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.000598, "corr_mean": 0.8732, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000593, "corr_mean": 0.8747, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.001146, "corr_mean": 0.8836, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000509, "corr_mean": 0.8709, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.001200, "corr_mean": 0.8728, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.000717, "corr_mean": 0.8691, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.002425, "corr_mean": 0.8598, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.000661, "corr_mean": 0.8673, "samples": 2486}
        ]
    },
    128: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.002347, "corr_mean": 0.9188, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.000453, "corr_mean": 0.9047, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000476, "corr_mean": 0.9034, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.000880, "corr_mean": 0.9115, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000400, "corr_mean": 0.8997, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.000877, "corr_mean": 0.9056, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.000535, "corr_mean": 0.9018, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.001911, "corr_mean": 0.8941, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.000500, "corr_mean": 0.9012, "samples": 2486}
        ]
    },
    256: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.001759, "corr_mean": 0.9340, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.000381, "corr_mean": 0.9209, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000411, "corr_mean": 0.9204, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.000659, "corr_mean": 0.9261, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000320, "corr_mean": 0.9169, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.000690, "corr_mean": 0.9228, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.000442, "corr_mean": 0.9190, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.001528, "corr_mean": 0.9114, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.000413, "corr_mean": 0.9181, "samples": 2486}
        ]
    },
    512: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.001620, "corr_mean": 0.9509, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.000291, "corr_mean": 0.9418, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000306, "corr_mean": 0.9404, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.000554, "corr_mean": 0.9451, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000256, "corr_mean": 0.9379, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.000510, "corr_mean": 0.9435, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.000337, "corr_mean": 0.9410, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.001220, "corr_mean": 0.9298, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.000307, "corr_mean": 0.9400, "samples": 2486}
        ]
    },
    1024: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.001303, "corr_mean": 0.9535, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.000270, "corr_mean": 0.9442, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000285, "corr_mean": 0.9426, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.000457, "corr_mean": 0.9472, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000232, "corr_mean": 0.9403, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.000481, "corr_mean": 0.9458, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.000310, "corr_mean": 0.9438, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.001129, "corr_mean": 0.9330, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.000292, "corr_mean": 0.9427, "samples": 2486}
        ]
    },
    2048: {
        "sessions": [
            {"session_id": 501559087, "mse_mean": 0.001522, "corr_mean": 0.9415, "samples": 2277},
            {"session_id": 501498760, "mse_mean": 0.000331, "corr_mean": 0.9303, "samples": 2282},
            {"session_id": 501836392, "mse_mean": 0.000331, "corr_mean": 0.9289, "samples": 2315},
            {"session_id": 501474098, "mse_mean": 0.000542, "corr_mean": 0.9348, "samples": 2114},
            {"session_id": 663488086, "mse_mean": 0.000283, "corr_mean": 0.9259, "samples": 2348},
            {"session_id": 647593956, "mse_mean": 0.000588, "corr_mean": 0.9326, "samples": 2486},
            {"session_id": 627823723, "mse_mean": 0.000382, "corr_mean": 0.9297, "samples": 2481},
            {"session_id": 569494121, "mse_mean": 0.001304, "corr_mean": 0.9217, "samples": 2486},
            {"session_id": 566523247, "mse_mean": 0.000362, "corr_mean": 0.9286, "samples": 2486}
        ]
    }
}

# ============================================================================
# DATI DUALI (Dual Neurons)
# ============================================================================

METRICS_DATA_DUAL = {
    16: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.00526, "corr_mean": 0.7354, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.001351, "corr_mean": 0.6647, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.001209, "corr_mean": 0.7149, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.00243, "corr_mean": 0.7054, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.001003, "corr_mean": 0.6932, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.002468, "corr_mean": 0.6518, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.001666, "corr_mean": 0.6304, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.004556, "corr_mean": 0.6431, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.001564, "corr_mean": 0.626, "samples": 2486}
        ]
    },
    32: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.004199, "corr_mean": 0.7672, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.001188, "corr_mean": 0.7026, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000967, "corr_mean": 0.7504, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.001872, "corr_mean": 0.7393, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000855, "corr_mean": 0.7308, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.002086, "corr_mean": 0.6917, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.001489, "corr_mean": 0.6705, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.003987, "corr_mean": 0.6785, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.001441, "corr_mean": 0.6656, "samples": 2486}
        ]
    },
    64: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.00351, "corr_mean": 0.8391, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.000887, "corr_mean": 0.8, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000811, "corr_mean": 0.8241, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.001428, "corr_mean": 0.8222, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000679, "corr_mean": 0.8114, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.001599, "corr_mean": 0.7975, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.001053, "corr_mean": 0.7869, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.003031, "corr_mean": 0.7866, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.001, "corr_mean": 0.7831, "samples": 2486}
        ]
    },
    128: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.003346, "corr_mean": 0.8953, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.000585, "corr_mean": 0.878, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.00064, "corr_mean": 0.8769, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.001239, "corr_mean": 0.8861, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000521, "corr_mean": 0.8728, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.001097, "corr_mean": 0.878, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.000739, "corr_mean": 0.8744, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.002407, "corr_mean": 0.8642, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.000628, "corr_mean": 0.8726, "samples": 2486}
        ]
    },
    256: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.003157, "corr_mean": 0.9194, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.000455, "corr_mean": 0.9073, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000496, "corr_mean": 0.9049, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.001006, "corr_mean": 0.9119, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000419, "corr_mean": 0.901, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.000982, "corr_mean": 0.9094, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.000534, "corr_mean": 0.9063, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.001968, "corr_mean": 0.8936, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.000481, "corr_mean": 0.9053, "samples": 2486}
        ]
    },
    512: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.002304, "corr_mean": 0.9347, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.000381, "corr_mean": 0.9241, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000408, "corr_mean": 0.9228, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.000821, "corr_mean": 0.9285, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000342, "corr_mean": 0.9196, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.000756, "corr_mean": 0.926, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.000453, "corr_mean": 0.9227, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.001513, "corr_mean": 0.9138, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.000397, "corr_mean": 0.9217, "samples": 2486}
        ]
    },
    1024: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.001818, "corr_mean": 0.9501, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.000292, "corr_mean": 0.9418, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000322, "corr_mean": 0.9408, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.000626, "corr_mean": 0.945, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000258, "corr_mean": 0.9384, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.000567, "corr_mean": 0.9435, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.000353, "corr_mean": 0.941, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.001221, "corr_mean": 0.9319, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.000308, "corr_mean": 0.9402, "samples": 2486}
        ]
    },
    2048: {
        "sessions": [
            {"id": 501559087, "mse_mean": 0.001383, "corr_mean": 0.9581, "samples": 2277},
            {"id": 501498760, "mse_mean": 0.000246, "corr_mean": 0.9502, "samples": 2282},
            {"id": 501836392, "mse_mean": 0.000263, "corr_mean": 0.9487, "samples": 2315},
            {"id": 501474098, "mse_mean": 0.000481, "corr_mean": 0.9528, "samples": 2114},
            {"id": 663488086, "mse_mean": 0.000215, "corr_mean": 0.9464, "samples": 2348},
            {"id": 647593956, "mse_mean": 0.000463, "corr_mean": 0.9519, "samples": 2486},
            {"id": 627823723, "mse_mean": 0.000284, "corr_mean": 0.9498, "samples": 2481},
            {"id": 569494121, "mse_mean": 0.001039, "corr_mean": 0.94, "samples": 2486},
            {"id": 566523247, "mse_mean": 0.000266, "corr_mean": 0.9491, "samples": 2486}
        ]
    }
}

# Available codebook sizes
CODEBOOK_SIZES = list(METRICS_DATA_NON_DUAL.keys())

# ============================================================================
# FUNCTIONS
# ============================================================================

def extract_session_metrics(data):
    """Extract correlation and MSE for each session"""
    sessions = data.get('sessions', [])
    
    correlations = []
    mses = []
    
    for session in sessions:
        correlations.append(session.get('corr_mean', np.nan))
        mses.append(session.get('mse_mean', np.nan))
    
    return correlations, mses


def aggregate_by_session_type(correlations, mses):
    """Aggregate metrics by session type (VISP, VISL, VISAM)"""
    aggregated = {}
    
    for session_type, indices in SESSION_TYPE_INDICES.items():
        type_corrs = [correlations[i] for i in indices if i < len(correlations)]
        type_mses = [mses[i] for i in indices if i < len(mses)]
        
        type_corrs = [c for c in type_corrs if not np.isnan(c)]
        type_mses = [m for m in type_mses if not np.isnan(m)]
        
        aggregated[session_type] = {
            'correlation_mean': np.mean(type_corrs) if type_corrs else np.nan,
            'correlation_std': np.std(type_corrs) if type_corrs else np.nan,
            'mse_mean': np.mean(type_mses) if type_mses else np.nan,
            'mse_std': np.std(type_mses) if type_mses else np.nan,
        }
    
    return aggregated


def collect_all_data(metrics_data, model_type):
    """Collect data for all codebook sizes for a specific model type"""
    
    data_by_type = {
        'VISP': {'codebook_sizes': [], 'correlations': [], 'corr_stds': [], 'mses': [], 'mse_stds': []},
        'VISL': {'codebook_sizes': [], 'correlations': [], 'corr_stds': [], 'mses': [], 'mse_stds': []},
        'VISAM': {'codebook_sizes': [], 'correlations': [], 'corr_stds': [], 'mses': [], 'mse_stds': []},
    }
    
    print(f"\n📊 Loading metrics for {model_type} model...")
    
    for size in CODEBOOK_SIZES:
        data = metrics_data.get(size)
        if data is None:
            continue
        
        correlations, mses = extract_session_metrics(data)
        if correlations is None or mses is None:
            continue
        
        aggregated = aggregate_by_session_type(correlations, mses)
        
        for session_type in ['VISP', 'VISL', 'VISAM']:
            if not np.isnan(aggregated[session_type]['correlation_mean']):
                data_by_type[session_type]['codebook_sizes'].append(size)
                data_by_type[session_type]['correlations'].append(aggregated[session_type]['correlation_mean'])
                data_by_type[session_type]['corr_stds'].append(aggregated[session_type]['correlation_std'])
                data_by_type[session_type]['mses'].append(aggregated[session_type]['mse_mean'])
                data_by_type[session_type]['mse_stds'].append(aggregated[session_type]['mse_std'])
    
    return data_by_type


def plot_comparison(data_non_dual, data_dual, output_dir):
    """Create comparison plots with 6 lines (3 session types × 2 model types)"""
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Colors for session types
    colors_base = {
        'VISP': '#e74c3c',    # Red
        'VISL': '#3498db',    # Blue
        'VISAM': '#2ecc71'    # Green
    }
    
    markers = {
        'non_dual': 'o',      # Circle for non-dual
        'dual': 's'           # Square for dual
    }
    
    # ========================================================================
    # PLOT 1: CORRELATION
    # ========================================================================
    ax1 = axes[0]
    
    for session_type in ['VISP', 'VISL', 'VISAM']:
        base_color = colors_base[session_type]
        
        # Non-Dual (solid line, lighter)
        data_nd = data_non_dual[session_type]
        if data_nd['codebook_sizes']:
            ax1.errorbar(
                data_nd['codebook_sizes'], data_nd['correlations'], 
                yerr=data_nd['corr_stds'],
                marker=markers['non_dual'], markersize=8,
                linewidth=2.5, linestyle='-', alpha=0.9,
                capsize=5, capthick=2,
                color=base_color,
                label=f'{session_type} (Non-Dual)'
            )
        
        # Dual (dashed line, darker)
        data_d = data_dual[session_type]
        if data_d['codebook_sizes']:
            # Make dual darker
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(base_color)
            darker_color = tuple(max(0, c * 0.7) for c in rgb)
            
            ax1.errorbar(
                data_d['codebook_sizes'], data_d['correlations'],
                yerr=data_d['corr_stds'],
                marker=markers['dual'], markersize=8,
                linewidth=2.5, linestyle='--', alpha=0.9,
                capsize=5, capthick=2,
                color=darker_color,
                label=f'{session_type} (Dual)'
            )
    
    ax1.set_xlabel('Codebook Size', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Correlation (mean ± std)', fontsize=14, fontweight='bold')
    ax1.set_title('Reconstruction Correlation\nDual vs Non-Dual Models', 
                  fontsize=15, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_xticks(CODEBOOK_SIZES)
    ax1.set_xticklabels([str(s) for s in CODEBOOK_SIZES], rotation=45)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10, loc='lower right', ncol=2)
    ax1.set_ylim([0.6, 1.0])
    
    # Reference lines
    ax1.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=0.90, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # ========================================================================
    # PLOT 2: MSE
    # ========================================================================
    ax2 = axes[1]
    
    for session_type in ['VISP', 'VISL', 'VISAM']:
        base_color = colors_base[session_type]
        
        # Non-Dual
        data_nd = data_non_dual[session_type]
        if data_nd['codebook_sizes']:
            ax2.errorbar(
                data_nd['codebook_sizes'], data_nd['mses'],
                yerr=data_nd['mse_stds'],
                marker=markers['non_dual'], markersize=8,
                linewidth=2.5, linestyle='-', alpha=0.9,
                capsize=5, capthick=2,
                color=base_color,
                label=f'{session_type} (Non-Dual)'
            )
        
        # Dual
        data_d = data_dual[session_type]
        if data_d['codebook_sizes']:
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(base_color)
            darker_color = tuple(max(0, c * 0.7) for c in rgb)
            
            ax2.errorbar(
                data_d['codebook_sizes'], data_d['mses'],
                yerr=data_d['mse_stds'],
                marker=markers['dual'], markersize=8,
                linewidth=2.5, linestyle='--', alpha=0.9,
                capsize=5, capthick=2,
                color=darker_color,
                label=f'{session_type} (Dual)'
            )
    
    ax2.set_xlabel('Codebook Size', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MSE (mean ± std)', fontsize=14, fontweight='bold')
    ax2.set_title('Reconstruction MSE\nDual vs Non-Dual Models', 
                  fontsize=15, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks(CODEBOOK_SIZES)
    ax2.set_xticklabels([str(s) for s in CODEBOOK_SIZES], rotation=45)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10, loc='upper right', ncol=2)
    
    # ========================================================================
    # OVERALL TITLE
    # ========================================================================
    plt.suptitle('VQ-VAE Performance Comparison: Dual vs Non-Dual Models\n' + 
                 '(Aggregated by Session Type: VISP, VISL, VISAM)', 
                 fontsize=17, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'comparison_dual_vs_non_dual.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Figure saved: {output_path}")
    
    plt.show()
    
    return fig


def print_comparison_table(data_non_dual, data_dual):
    """Print comparison table"""
    
    print("\n" + "="*100)
    print("📊 COMPARISON TABLE: Dual vs Non-Dual Models")
    print("="*100)
    
    print(f"\n{'Codebook':<10} {'Session':<8} {'Model':<12} {'Correlation':<22} {'MSE':<22}")
    print(f"{'Size':<10} {'Type':<8} {'Type':<12} {'(mean ± std)':<22} {'(mean ± std)':<22}")
    print("-" * 100)
    
    for size in CODEBOOK_SIZES:
        for session_type in ['VISP', 'VISL', 'VISAM']:
            # Non-Dual
            data_nd = data_non_dual[session_type]
            if size in data_nd['codebook_sizes']:
                idx = data_nd['codebook_sizes'].index(size)
                corr_nd = data_nd['correlations'][idx]
                corr_std_nd = data_nd['corr_stds'][idx]
                mse_nd = data_nd['mses'][idx]
                mse_std_nd = data_nd['mse_stds'][idx]
                
                print(f"{size:<10} {session_type:<8} {'Non-Dual':<12} "
                      f"{corr_nd:.4f} ± {corr_std_nd:.4f}       "
                      f"{mse_nd:.6f} ± {mse_std_nd:.6f}")
            
            # Dual
            data_d = data_dual[session_type]
            if size in data_d['codebook_sizes']:
                idx = data_d['codebook_sizes'].index(size)
                corr_d = data_d['correlations'][idx]
                corr_std_d = data_d['corr_stds'][idx]
                mse_d = data_d['mses'][idx]
                mse_std_d = data_d['mse_stds'][idx]
                
                print(f"{'':<10} {'':<8} {'Dual':<12} "
                      f"{corr_d:.4f} ± {corr_std_d:.4f}       "
                      f"{mse_d:.6f} ± {mse_std_d:.6f}")
        
        if size != CODEBOOK_SIZES[-1]:
            print("-" * 100)
    
    print("="*100)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*100)
    print("📈 DUAL vs NON-DUAL MODEL COMPARISON ANALYSIS")
    print("="*100)
    
    # Collect data for both model types
    data_non_dual = collect_all_data(METRICS_DATA_NON_DUAL, "Non-Dual")
    data_dual = collect_all_data(METRICS_DATA_DUAL, "Dual")
    
    # Print comparison table
    print_comparison_table(data_non_dual, data_dual)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    print("\n📊 Creating comparison plots...")
    fig = plot_comparison(data_non_dual, data_dual, OUTPUT_DIR)
    
    print("\n" + "="*100)
    print("✅ Comparison analysis complete!")
    print("="*100)


if __name__ == "__main__":
    main()