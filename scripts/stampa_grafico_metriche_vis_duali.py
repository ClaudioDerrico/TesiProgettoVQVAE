#!/usr/bin/env python3
"""
Plot codebook size vs metrics (correlation and MSE)
Aggregated by session type: VISP, VISL, VISAM
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

OUTPUT_DIR = Path(r"C:\Users\Claudio\Desktop\TesiProgettoVQVAE\metriche_vis_duali")

# Mapping session types based on position in results
# First 3 sessions: VISP
# Next 3 sessions: VISL  
# Last 3 sessions: VISAM
SESSION_TYPE_INDICES = {
    'VISP': [0, 1, 2],
    'VISL': [3, 4, 5],
    'VISAM': [6, 7, 8]
}

# JSON data embedded directly
METRICS_DATA = {
    16: {
  "model_config": {
    "num_neurons": 1,
    "num_hiddens": 128,
    "num_residual_layers": 3,
    "num_residual_hiddens": 64,
    "num_embeddings": 16,
    "embedding_dim": 32,
    "commitment_cost": 0.25,
    "dropout_rate": 0.1,
    "use_quantizer": True
  },
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
  "model_config": {
    "num_neurons": 1,
    "num_hiddens": 128,
    "num_residual_layers": 3,
    "num_residual_hiddens": 64,
    "num_embeddings": 32,
    "embedding_dim": 32,
    "commitment_cost": 0.25,
    "dropout_rate": 0.1,
    "use_quantizer": True
  },
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
  "model_config": {
    "num_neurons": 1,
    "num_hiddens": 128,
    "num_residual_layers": 3,
    "num_residual_hiddens": 64,
    "num_embeddings": 64,
    "embedding_dim": 32,
    "commitment_cost": 0.25,
    "dropout_rate": 0.1,
    "use_quantizer": True
  },
  "session_results": {
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

}
},
    128: {
  "model_config": {
    "num_neurons": 1,
    "num_hiddens": 128,
    "num_residual_layers": 3,
    "num_residual_hiddens": 64,
    "num_embeddings": 128,
    "embedding_dim": 32,
    "commitment_cost": 0.25,
    "dropout_rate": 0.1,
    "use_quantizer": True
  },
  "session_results": [
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
  "model_config": {
    "num_neurons": 1,
    "num_hiddens": 128,
    "num_residual_layers": 3,
    "num_residual_hiddens": 64,
    "num_embeddings": 256,
    "embedding_dim": 64,
    "commitment_cost": 0.25,
    "dropout_rate": 0.1,
    "use_quantizer": True
  },
  "session_results": [
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
  "model_config": {
    "num_neurons": 1,
    "num_hiddens": 128,
    "num_residual_layers": 3,
    "num_residual_hiddens": 64,
    "num_embeddings": 512,
    "embedding_dim": 128,
    "commitment_cost": 0.25,
    "dropout_rate": 0.1,
    "use_quantizer": True
  },
  "session_results":[
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
  "model_config": {
    "num_neurons": 1,
    "num_hiddens": 128,
    "num_residual_layers": 3,
    "num_residual_hiddens": 64,
    "num_embeddings": 1024,
    "embedding_dim": 128,
    "commitment_cost": 0.25,
    "dropout_rate": 0.1,
    "use_quantizer": True
  },
  "session_results": [
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
  "model_config": {
  "num_neurons": 1,
  "num_hiddens": 128,
  "num_residual_layers": 3,
  "num_residual_hiddens": 64,
  "num_embeddings": 2048,
  "embedding_dim": 128,
  "commitment_cost": 0.25,
  "dropout_rate": 0.1,
  "use_quantizer": True
},
  "session_results":[
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
CODEBOOK_SIZES = list(METRICS_DATA.keys())

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_metrics_data(codebook_size):
    """Get metrics data for a specific codebook size from embedded data"""
    if codebook_size not in METRICS_DATA:
        print(f"⚠️  Codebook size {codebook_size} not found in data")
        return None
    
    return METRICS_DATA[codebook_size]


def extract_session_metrics(data):
    """
    Extract correlation and MSE for each session from the JSON data
    Returns two lists: correlations and mses
    """
    # Try different possible keys for session results
    sessions = None
    
    if 'session_results' in data:
        # Check if it's a list or dict
        if isinstance(data['session_results'], list):
            sessions = data['session_results']
        elif isinstance(data['session_results'], dict) and 'sessions' in data['session_results']:
            sessions = data['session_results']['sessions']
    elif 'sessions' in data:
        sessions = data['sessions']
    
    if sessions is None:
        print(f"❌ Unknown data structure: {data.keys()}")
        return None, None
    
    correlations = []
    mses = []
    
    for session in sessions:
        # Handle different key names
        if 'corr_mean' in session:
            correlations.append(session['corr_mean'])
        elif 'correlation_mean' in session:
            correlations.append(session['correlation_mean'])
        else:
            print(f"⚠️  No correlation key found in session: {session.keys()}")
            correlations.append(np.nan)
        
        if 'mse_mean' in session:
            mses.append(session['mse_mean'])
        elif 'mse' in session:
            mses.append(session['mse'])
        else:
            print(f"⚠️  No MSE key found in session: {session.keys()}")
            mses.append(np.nan)
    
    return correlations, mses


def aggregate_by_session_type(correlations, mses):
    """
    Aggregate metrics by session type (VISP, VISL, VISAM)
    Returns dict with mean values for each type
    """
    aggregated = {}
    
    for session_type, indices in SESSION_TYPE_INDICES.items():
        # Extract values for this session type
        type_corrs = [correlations[i] for i in indices if i < len(correlations)]
        type_mses = [mses[i] for i in indices if i < len(mses)]
        
        # Calculate means (filtering out NaNs)
        type_corrs = [c for c in type_corrs if not np.isnan(c)]
        type_mses = [m for m in type_mses if not np.isnan(m)]
        
        aggregated[session_type] = {
            'correlation_mean': np.mean(type_corrs) if type_corrs else np.nan,
            'correlation_std': np.std(type_corrs) if type_corrs else np.nan,
            'mse_mean': np.mean(type_mses) if type_mses else np.nan,
            'mse_std': np.std(type_mses) if type_mses else np.nan,
        }
    
    return aggregated


def collect_all_data():
    """
    Collect data for all codebook sizes
    Returns dict organized by session type and metric
    """
    
    # Initialize data structure
    data_by_type = {
        'VISP': {'codebook_sizes': [], 'correlations': [], 'corr_stds': [], 'mses': [], 'mse_stds': []},
        'VISL': {'codebook_sizes': [], 'correlations': [], 'corr_stds': [], 'mses': [], 'mse_stds': []},
        'VISAM': {'codebook_sizes': [], 'correlations': [], 'corr_stds': [], 'mses': [], 'mse_stds': []},
    }
    
    print("📊 Loading metrics from all codebook sizes...")
    
    for size in CODEBOOK_SIZES:
        print(f"\n   Processing codebook size: {size}")
        
        # Load metrics data
        data = load_metrics_data(size)
        if data is None:
            print(f"      ⚠️  Skipping {size} (data not found)")
            continue
        
        # Extract session-level metrics
        correlations, mses = extract_session_metrics(data)
        if correlations is None or mses is None:
            print(f"      ⚠️  Skipping {size} (could not extract metrics)")
            continue
        
        print(f"      ✅ Extracted {len(correlations)} sessions")
        
        # Aggregate by session type
        aggregated = aggregate_by_session_type(correlations, mses)
        
        # Store in data structure
        for session_type in ['VISP', 'VISL', 'VISAM']:
            if not np.isnan(aggregated[session_type]['correlation_mean']):
                data_by_type[session_type]['codebook_sizes'].append(size)
                data_by_type[session_type]['correlations'].append(aggregated[session_type]['correlation_mean'])
                data_by_type[session_type]['corr_stds'].append(aggregated[session_type]['correlation_std'])
                data_by_type[session_type]['mses'].append(aggregated[session_type]['mse_mean'])
                data_by_type[session_type]['mse_stds'].append(aggregated[session_type]['mse_std'])
                
                print(f"         {session_type}: corr={aggregated[session_type]['correlation_mean']:.4f}, "
                      f"mse={aggregated[session_type]['mse_mean']:.6f}")
    
    return data_by_type


def plot_metrics_vs_codebook_size(data_by_type, output_dir):
    """
    Create 2 subplots: correlation and MSE vs codebook size
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors for each session type
    colors = {
        'VISP': '#e74c3c',    # Red
        'VISL': '#3498db',    # Blue
        'VISAM': '#2ecc71'    # Green
    }
    
    markers = {
        'VISP': 'o',
        'VISL': 's',
        'VISAM': '^'
    }
    
    # ========================================================================
    # PLOT 1: CORRELATION vs CODEBOOK SIZE
    # ========================================================================
    ax1 = axes[0]
    
    for session_type in ['VISP', 'VISL', 'VISAM']:
        data = data_by_type[session_type]
        
        if not data['codebook_sizes']:
            continue
        
        sizes = np.array(data['codebook_sizes'])
        corrs = np.array(data['correlations'])
        corr_stds = np.array(data['corr_stds'])
        
        # Plot with error bars
        ax1.errorbar(
            sizes, corrs, yerr=corr_stds,
            marker=markers[session_type], 
            markersize=8,
            linewidth=2.5,
            capsize=5,
            capthick=2,
            alpha=0.8,
            color=colors[session_type],
            label=session_type
        )
    
    ax1.set_xlabel('Codebook Size (num_embeddings)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Correlation (mean ± std)', fontsize=13, fontweight='bold')
    ax1.set_title('Reconstruction Correlation vs Codebook Size', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_xticks(CODEBOOK_SIZES)
    ax1.set_xticklabels([str(s) for s in CODEBOOK_SIZES])
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.set_ylim([0.6, 1.0])
    
    # Add horizontal line at 0.95 (excellent correlation threshold)
    ax1.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (0.95)')
    ax1.axhline(y=0.90, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Good (0.90)')
    
    # ========================================================================
    # PLOT 2: MSE vs CODEBOOK SIZE
    # ========================================================================
    ax2 = axes[1]
    
    for session_type in ['VISP', 'VISL', 'VISAM']:
        data = data_by_type[session_type]
        
        if not data['codebook_sizes']:
            continue
        
        sizes = np.array(data['codebook_sizes'])
        mses = np.array(data['mses'])
        mse_stds = np.array(data['mse_stds'])
        
        # Plot with error bars
        ax2.errorbar(
            sizes, mses, yerr=mse_stds,
            marker=markers[session_type],
            markersize=8,
            linewidth=2.5,
            capsize=5,
            capthick=2,
            alpha=0.8,
            color=colors[session_type],
            label=session_type
        )
    
    ax2.set_xlabel('Codebook Size (num_embeddings)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('MSE (mean ± std)', fontsize=13, fontweight='bold')
    ax2.set_title('Reconstruction MSE vs Codebook Size', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks(CODEBOOK_SIZES)
    ax2.set_xticklabels([str(s) for s in CODEBOOK_SIZES])
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=11, loc='upper right')
    
    # ========================================================================
    # OVERALL TITLE AND LAYOUT
    # ========================================================================
    plt.suptitle('VQ-VAE Performance vs Codebook Size\n(Aggregated by Session Type)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'codebook_size_vs_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Figure saved: {output_path}")
    
    plt.show()
    
    return fig


def print_summary_table(data_by_type):
    """Print a summary table of results"""
    
    print("\n" + "="*80)
    print("📊 SUMMARY TABLE: Performance by Codebook Size and Session Type")
    print("="*80)
    
    # Header
    print(f"\n{'Codebook':<12} {'Session':<10} {'Correlation':<20} {'MSE':<20}")
    print(f"{'Size':<12} {'Type':<10} {'(mean ± std)':<20} {'(mean ± std)':<20}")
    print("-" * 80)
    
    # Data rows
    for size in CODEBOOK_SIZES:
        for session_type in ['VISP', 'VISL', 'VISAM']:
            data = data_by_type[session_type]
            
            if size in data['codebook_sizes']:
                idx = data['codebook_sizes'].index(size)
                corr_mean = data['correlations'][idx]
                corr_std = data['corr_stds'][idx]
                mse_mean = data['mses'][idx]
                mse_std = data['mse_stds'][idx]
                
                print(f"{size:<12} {session_type:<10} "
                      f"{corr_mean:.4f} ± {corr_std:.4f}     "
                      f"{mse_mean:.6f} ± {mse_std:.6f}")
        
        if size != CODEBOOK_SIZES[-1]:
            print("-" * 80)
    
    print("="*80)
    
    # Best performance summary
    print("\n🏆 BEST PERFORMANCE BY SESSION TYPE:")
    for session_type in ['VISP', 'VISL', 'VISAM']:
        data = data_by_type[session_type]
        
        if not data['correlations']:
            continue
        
        best_corr_idx = np.argmax(data['correlations'])
        best_corr_size = data['codebook_sizes'][best_corr_idx]
        best_corr = data['correlations'][best_corr_idx]
        
        best_mse_idx = np.argmin(data['mses'])
        best_mse_size = data['codebook_sizes'][best_mse_idx]
        best_mse = data['mses'][best_mse_idx]
        
        print(f"\n   {session_type}:")
        print(f"      Best Correlation: {best_corr:.4f} (codebook size: {best_corr_size})")
        print(f"      Best MSE: {best_mse:.6f} (codebook size: {best_mse_size})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("📈 CODEBOOK SIZE vs PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Collect all data
    data_by_type = collect_all_data()
    
    # Check if we have data
    has_data = any(len(data['codebook_sizes']) > 0 for data in data_by_type.values())
    
    if not has_data:
        print("\n❌ No data found!")
        return
    
    # Print summary table
    print_summary_table(data_by_type)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    print("\n📊 Creating plots...")
    fig = plot_metrics_vs_codebook_size(data_by_type, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()