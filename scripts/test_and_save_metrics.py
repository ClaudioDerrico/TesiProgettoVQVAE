#!/usr/bin/env python3
"""
Script per testare un modello SINGLE NEURON su sessioni e salvare metriche in JSON.

Salva corr_mean e mse_mean per ogni sessione in formato JSON.

Uso:
    python scripts/test_and_save_metrics.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from datasets.calcium import UnifiedMultiSessionDataset

from models.vqvae import CalciumVQVAE
from datasets.calcium import TEST_SESSION_IDS

# ============================================================================
# CONFIGURAZIONE - MODIFICA QUI
# ============================================================================

CHECKPOINT_PATH = "results/best_single_neuron_model_2048.pth"  # ✅ MODIFICA QUESTO

MODEL_CONFIG = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 2048, 
    'embedding_dim': 128,
    'commitment_cost': 0.25,
    'dropout_rate': 0.1,
    'use_quantizer': True,
}

# Sessioni da testare (modifica se necessario)
SESSIONS_TO_TEST = TEST_SESSION_IDS

# Parametri dataset
BATCH_SIZE = 64
WINDOW_SIZE = 60
STRIDE = 50

# ============================================================================
# FUNZIONI
# ============================================================================

def load_trained_model(checkpoint_path, config, device):
    """Carica il modello dai pesi salvati"""
    print(f"🔄 Caricando modello da: {checkpoint_path}")
    
    model = CalciumVQVAE(
        num_neurons=config['num_neurons'],
        num_hiddens=config['num_hiddens'],
        num_residual_layers=config['num_residual_layers'],
        num_residual_hiddens=config['num_residual_hiddens'],
        num_embeddings=config['num_embeddings'],
        embedding_dim=config['embedding_dim'],
        commitment_cost=config['commitment_cost'],
        dropout_rate=config.get('dropout_rate', 0.0),
        use_quantizer=config.get('use_quantizer', True)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Pesi caricati")
        
        if 'best_correlation' in checkpoint:
            print(f"📊 Best training correlation: {checkpoint['best_correlation']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Pesi caricati")
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_session(model, session_id, device, window_size, stride):
    """
    Valuta il modello su una specifica sessione e restituisce metriche
    """
    
    print(f"\n{'='*70}")
    print(f"📊 Testando sessione: {session_id}")
    print(f"{'='*70}")
    
    try:
        # Carica dataset
        dataset = UnifiedMultiSessionDataset(
            session_ids=[session_id],
            window_size=window_size,
            stride=stride,
            min_neurons=1,  # 1 NEURONE
            mode='train'    # Train mode = random selection (best neuron)
        )
        
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        total_samples = len(dataset)
        print(f"✅ Dataset caricato: {total_samples} campioni")
        
        all_mse = []
        all_corr = []
        
        samples_processed = 0
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                neural_data = batch_data.to(device)
                
                # Forward pass
                vq_loss, neural_recon, perplexity, _, _ = model(neural_data)
                
                # Converti in numpy
                original = neural_data.cpu().numpy()
                reconstructed = neural_recon.cpu().numpy()
                
                # Calcola metriche per TUTTI i campioni del batch
                for i in range(original.shape[0]):
                    sample_orig = original[i, 0, :]  # Shape: (60,)
                    sample_recon = reconstructed[i, 0, :]
                    
                    mse = np.mean((sample_orig - sample_recon) ** 2)
                    
                    if np.std(sample_orig) > 1e-8 and np.std(sample_recon) > 1e-8:
                        corr, _ = pearsonr(sample_orig, sample_recon)
                        corr = corr if np.isfinite(corr) else 0
                    else:
                        corr = 0
                    
                    all_mse.append(mse)
                    all_corr.append(corr)
                    samples_processed += 1
                
                # Print ogni 50 batch
                if (batch_idx + 1) % 50 == 0:
                    print(f"   Processati {batch_idx + 1} batch, {samples_processed} campioni...")
        
        # Calcola metriche finali
        mse_mean = float(np.mean(all_mse))
        corr_mean = float(np.mean(all_corr))
        
        print(f"\n📈 RISULTATI:")
        print(f"   Campioni analizzati: {samples_processed}")
        print(f"   MSE mean:  {mse_mean:.6f}")
        print(f"   Corr mean: {corr_mean:.4f}")
        
        return {
            'session_id': int(session_id),
            'num_samples': samples_processed,
            'mse_mean': mse_mean,
            'corr_mean': corr_mean,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'session_id': int(session_id),
            'success': False,
            'error': str(e)
        }


def save_results_to_json(results, checkpoint_path):
    """
    Salva i risultati in JSON:
    - model_config indentato leggibile
    - session_results: ogni sessione su una singola riga
    """
    output_dir = Path('metrics_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(checkpoint_path).stem
    filename = f'{checkpoint_name}_metrics.json'
    output_path = output_dir / filename

    # Prepara sessioni
    sessions = []
    for r in results:
        if r.get('success'):
            sessions.append({
                "session_id": int(r["session_id"]),
                "mse_mean": round(float(r["mse_mean"]), 6),
                "corr_mean": round(float(r["corr_mean"]), 4),
                "samples": int(r["num_samples"])
            })

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('{\n')
        # model_config leggibile
        f.write(f'  "model_config": {json.dumps(MODEL_CONFIG, indent=2, ensure_ascii=False)},\n')
        # session_results flat
        f.write('  "session_results": [\n')
        for i, s in enumerate(sessions):
            line = f'    {json.dumps(s, ensure_ascii=False)}'
            if i < len(sessions) - 1:
                line += ','
            line += '\n'
            f.write(line)
        f.write('  ]\n')
        f.write('}\n')

    print(f"\n💾 Risultati salvati in JSON:")
    print(f"   {output_path}")

    return output_path



# ============================================================================
# MAIN
# ============================================================================

def main():
    print("🎯 TEST CHECKPOINT E SALVATAGGIO METRICHE IN JSON")
    print("="*70)
    print(f"📁 Checkpoint: {CHECKPOINT_PATH}")
    print(f"🔬 Sessioni da testare: {len(SESSIONS_TO_TEST)}")
    print(f"   {SESSIONS_TO_TEST}")
    print(f"📐 Parametri: 1 neurone, window={WINDOW_SIZE}, stride={STRIDE}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Verifica esistenza checkpoint
    if not Path(CHECKPOINT_PATH).exists():
        print(f"\n❌ ERRORE: Checkpoint non trovato: {CHECKPOINT_PATH}")
        print(f"   Modifica la variabile CHECKPOINT_PATH nello script!")
        return
    
    # Carica modello
    model = load_trained_model(CHECKPOINT_PATH, MODEL_CONFIG, device)
    
    # Testa su tutte le sessioni
    all_results = []
    
    for session_id in SESSIONS_TO_TEST:
        result = evaluate_session(model, session_id, device, WINDOW_SIZE, STRIDE)
        all_results.append(result)
    
    # Salva risultati in JSON
    json_path = save_results_to_json(all_results, CHECKPOINT_PATH)
    
    # Summary finale
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    successful_results = [r for r in all_results if r.get('success')]
    
    if successful_results:
        print(f"\n✅ Sessioni testate con successo: {len(successful_results)}/{len(all_results)}")
        
        # Calcola medie
        avg_mse = np.mean([r['mse_mean'] for r in successful_results])
        avg_corr = np.mean([r['corr_mean'] for r in successful_results])
        
        print(f"\n📊 Metriche medie su {len(successful_results)} sessioni:")
        print(f"   MSE mean:  {avg_mse:.6f}")
        print(f"   Corr mean: {avg_corr:.4f}")
        
        # Tabella risultati
        print(f"\n📋 Risultati per sessione:")
        print(f"{'Session ID':<15} {'MSE Mean':<15} {'Corr Mean':<15} {'Samples':<10}")
        print("-" * 55)
        
        for r in successful_results:
            print(f"{r['session_id']:<15} {r['mse_mean']:<15.6f} {r['corr_mean']:<15.4f} {r['num_samples']:<10}")
    else:
        print(f"\n❌ Nessuna sessione testata con successo")
    
    print(f"\n✅ Risultati salvati in: {json_path}")
    print("="*70)


if __name__ == "__main__":
    main()