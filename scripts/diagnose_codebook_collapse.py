#!/usr/bin/env python3
"""
🔬 DIAGNOSI CODEBOOK COLLAPSE

Questo script verifica SE e PERCHÉ il codebook è collassato.

Analizza:
1. ✅ Quanti codici vengono REALMENTE usati
2. ✅ Distribuzione degli encoding indices
3. ✅ Distanze tra input encoder e codebook embeddings
4. ✅ Valori di VQ loss e commitment cost durante forward
5. ✅ Varianza degli embeddings del codebook
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from models.vqvae import CalciumVQVAE
from datasets.calcium import create_unified_dataloaders, TRAINING_SESSION_IDS

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CHECKPOINT_PATH = "results/best_single_neuron_model_1024_pt2.pth"

MODEL_CONFIG = {
    'num_neurons': 1,
    'num_hiddens': 128,
    'num_residual_layers': 3,
    'num_residual_hiddens': 64,
    'num_embeddings': 1024,
    'embedding_dim': 128,
    'commitment_cost': 0.5,
    'dropout_rate': 0.0,
    'use_quantizer': True,
}

BATCH_SIZE = 64
NUM_BATCHES_TO_TEST = 20  # Test su 20 batch

# ============================================================================
# FUNZIONI
# ============================================================================

def load_model(checkpoint_path, device):
    """Carica modello"""
    print(f"🔄 Caricando: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    num_embeddings = state_dict['vector_quantization.embedding.weight'].shape[0]
    
    model = CalciumVQVAE(
        num_neurons=MODEL_CONFIG['num_neurons'],
        num_hiddens=MODEL_CONFIG['num_hiddens'],
        num_residual_layers=MODEL_CONFIG['num_residual_layers'],
        num_residual_hiddens=MODEL_CONFIG['num_residual_hiddens'],
        num_embeddings=num_embeddings,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        commitment_cost=MODEL_CONFIG['commitment_cost'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        use_quantizer=MODEL_CONFIG['use_quantizer']
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, num_embeddings


def diagnose_quantization(model, dataloader, device, num_batches):
    """
    Diagnosi COMPLETA del processo di quantizzazione
    """
    
    print("\n" + "="*70)
    print("🔬 DIAGNOSI QUANTIZZAZIONE")
    print("="*70)
    
    model.eval()
    
    # Collectors
    all_encoding_indices = []
    all_vq_losses = []
    all_perplexities = []
    all_encoder_outputs = []
    
    print(f"\n📊 Analizzando {num_batches} batch...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            neural_data = batch_data.to(device)
            
            # Forward pass
            vq_loss, recon, perplexity, encodings, encoding_indices = model(neural_data)
            
            # Raccogli dati
            all_vq_losses.append(vq_loss.item())
            all_perplexities.append(perplexity.item())
            
            if encoding_indices is not None:
                indices_flat = encoding_indices.flatten().cpu().numpy()
                all_encoding_indices.extend(indices_flat.tolist())
            
            # Raccogli output encoder (pre-quantization)
            z = model.encoder(neural_data)
            z = model.pre_quantization_conv(z)
            all_encoder_outputs.append(z.cpu().numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"   Batch {batch_idx + 1}/{num_batches}...")
    
    # ========================================================================
    # ANALISI 1: ENCODING INDICES
    # ========================================================================
    print("\n" + "="*70)
    print("📊 ANALISI 1: ENCODING INDICES")
    print("="*70)
    
    code_counter = Counter(all_encoding_indices)
    num_unique_codes = len(code_counter)
    total_indices = len(all_encoding_indices)
    
    print(f"\n✅ Codici unici usati: {num_unique_codes} / {MODEL_CONFIG['num_embeddings']}")
    print(f"   Percentuale: {100 * num_unique_codes / MODEL_CONFIG['num_embeddings']:.2f}%")
    print(f"   Total indices: {total_indices}")
    
    # Top 10 codici più usati
    most_common = code_counter.most_common(10)
    print(f"\n🔝 TOP 10 codici più usati:")
    for rank, (code_idx, count) in enumerate(most_common, 1):
        percentage = 100 * count / total_indices
        print(f"   {rank:2d}. Codice {code_idx:4d}: {count:6d} volte ({percentage:5.2f}%)")
    
    # ⚠️ DIAGNOSI COLLAPSE
    if num_unique_codes <= 5:
        print(f"\n❌ GRAVE CODEBOOK COLLAPSE!")
        print(f"   Solo {num_unique_codes} codici usati su {MODEL_CONFIG['num_embeddings']}")
        print(f"   Il modello è sostanzialmente un autoencoder senza VQ")
        
        # Controlla se c'è dominanza
        top_code_pct = 100 * most_common[0][1] / total_indices
        if top_code_pct > 90:
            print(f"\n❌ DOMINANZA ASSOLUTA!")
            print(f"   Codice {most_common[0][0]} usato {top_code_pct:.1f}% delle volte")
    
    # ========================================================================
    # ANALISI 2: VQ LOSS & PERPLEXITY
    # ========================================================================
    print("\n" + "="*70)
    print("📊 ANALISI 2: VQ LOSS & PERPLEXITY")
    print("="*70)
    
    avg_vq_loss = np.mean(all_vq_losses)
    avg_perplexity = np.mean(all_perplexities)
    
    print(f"\n✅ VQ Loss (mean): {avg_vq_loss:.6f}")
    print(f"✅ Perplexity (mean): {avg_perplexity:.2f}")
    
    # Perplexity teorica massima
    max_perplexity = MODEL_CONFIG['num_embeddings']
    perplexity_ratio = avg_perplexity / max_perplexity
    
    print(f"\n📐 Perplexity ratio: {perplexity_ratio:.4f}")
    print(f"   Max possibile: {max_perplexity}")
    print(f"   Attuale: {avg_perplexity:.1f}")
    
    if perplexity_ratio < 0.01:
        print(f"\n❌ PERPLEXITY TROPPO BASSA!")
        print(f"   Indica codebook collapse severo")
    elif perplexity_ratio < 0.05:
        print(f"\n⚠️  PERPLEXITY BASSA")
        print(f"   Possibile codebook collapse parziale")
    else:
        print(f"\n✅ Perplexity OK")
    
    # ========================================================================
    # ANALISI 3: ENCODER OUTPUT
    # ========================================================================
    print("\n" + "="*70)
    print("📊 ANALISI 3: ENCODER OUTPUT (pre-quantization)")
    print("="*70)
    
    encoder_outputs = np.concatenate(all_encoder_outputs, axis=0)
    
    # Shape: (batch, embedding_dim, time)
    encoder_flat = encoder_outputs.reshape(-1, encoder_outputs.shape[1])  # (batch*time, embedding_dim)
    
    print(f"\n✅ Encoder output shape: {encoder_outputs.shape}")
    print(f"   Flattened: {encoder_flat.shape}")
    
    # Statistiche
    encoder_mean = np.mean(encoder_flat, axis=0)  # (embedding_dim,)
    encoder_std = np.std(encoder_flat, axis=0)
    
    print(f"\n📐 Statistiche per dimensione embedding:")
    print(f"   Mean (avg across dims): {np.mean(encoder_mean):.4f}")
    print(f"   Std (avg across dims): {np.mean(encoder_std):.4f}")
    print(f"   Min std: {np.min(encoder_std):.4f}")
    print(f"   Max std: {np.max(encoder_std):.4f}")
    
    # ⚠️ DIAGNOSI VARIANZA
    if np.mean(encoder_std) < 0.1:
        print(f"\n❌ VARIANZA TROPPO BASSA!")
        print(f"   L'encoder produce output quasi costanti")
        print(f"   → Il VQ non può distinguere tra pattern diversi")
    
    # ========================================================================
    # ANALISI 4: CODEBOOK EMBEDDINGS
    # ========================================================================
    print("\n" + "="*70)
    print("📊 ANALISI 4: CODEBOOK EMBEDDINGS")
    print("="*70)
    
    codebook_embeddings = model.vector_quantization.embedding.weight.data.cpu().numpy()
    print(f"\n✅ Codebook shape: {codebook_embeddings.shape}")
    
    # Statistiche codebook
    cb_mean = np.mean(codebook_embeddings, axis=0)
    cb_std = np.std(codebook_embeddings, axis=0)
    
    print(f"\n📐 Statistiche codebook:")
    print(f"   Mean (avg): {np.mean(cb_mean):.4f}")
    print(f"   Std (avg): {np.mean(cb_std):.4f}")
    
    # Distanze tra embeddings
    from scipy.spatial.distance import pdist
    
    # Campiona 100 embeddings per velocità
    sample_size = min(100, codebook_embeddings.shape[0])
    sample_indices = np.random.choice(codebook_embeddings.shape[0], sample_size, replace=False)
    sampled_embeddings = codebook_embeddings[sample_indices]
    
    distances = pdist(sampled_embeddings, metric='euclidean')
    
    print(f"\n📏 Distanze tra embeddings (campione di {sample_size}):")
    print(f"   Mean distance: {np.mean(distances):.4f}")
    print(f"   Min distance: {np.min(distances):.4f}")
    print(f"   Max distance: {np.max(distances):.4f}")
    
    if np.min(distances) < 0.01:
        print(f"\n⚠️  EMBEDDINGS TROPPO VICINI!")
        print(f"   Alcuni embeddings sono quasi identici")
    
    # ========================================================================
    # ANALISI 5: DISTANZE ENCODER → CODEBOOK
    # ========================================================================
    print("\n" + "="*70)
    print("📊 ANALISI 5: DISTANZE ENCODER → CODEBOOK")
    print("="*70)
    
    # Prendi un campione di encoder outputs
    sample_encoder = encoder_flat[:1000]  # Prime 1000
    
    # Calcola distanze verso TUTTI i codebook embeddings
    distances_to_codebook = []
    
    for enc_vector in sample_encoder[:100]:  # Limita a 100 per velocità
        # Distanza L2 verso tutti gli embeddings
        dists = np.linalg.norm(codebook_embeddings - enc_vector, axis=1)
        distances_to_codebook.append(dists)
    
    distances_to_codebook = np.array(distances_to_codebook)  # (100, num_embeddings)
    
    # Trova il codice più vicino per ogni encoder output
    closest_codes = np.argmin(distances_to_codebook, axis=1)
    min_distances = np.min(distances_to_codebook, axis=1)
    
    print(f"\n✅ Analizzati 100 encoder outputs")
    print(f"   Mean distance to closest code: {np.mean(min_distances):.4f}")
    print(f"   Std distance: {np.std(min_distances):.4f}")
    
    # Distribuzione dei codici più vicini
    closest_counter = Counter(closest_codes)
    print(f"\n🔝 Codici più vicini agli encoder outputs:")
    for code_idx, count in closest_counter.most_common(5):
        print(f"   Codice {code_idx}: {count} volte")
    
    # ========================================================================
    # DIAGNOSI FINALE
    # ========================================================================
    print("\n" + "="*70)
    print("🎯 DIAGNOSI FINALE")
    print("="*70)
    
    issues = []
    
    if num_unique_codes <= 5:
        issues.append("❌ CODEBOOK COLLAPSE SEVERO (≤5 codici usati)")
    
    if perplexity_ratio < 0.01:
        issues.append("❌ PERPLEXITY TROPPO BASSA")
    
    if np.mean(encoder_std) < 0.1:
        issues.append("❌ ENCODER OUTPUT TROPPO UNIFORME")
    
    if avg_vq_loss < 0.001:
        issues.append("❌ VQ LOSS TROPPO BASSO (commitment cost insufficiente?)")
    
    if issues:
        print("\n⚠️  PROBLEMI RILEVATI:\n")
        for issue in issues:
            print(f"   {issue}")
        
        print("\n💡 POSSIBILI CAUSE:")
        print("   1. Commitment cost troppo BASSO (0.5 potrebbe non bastare)")
        print("   2. VQ weight scheduling troppo conservativo")
        print("   3. Encoder non addestrato abbastanza (produce output uniformi)")
        print("   4. Learning rate troppo alta all'inizio")
        
        print("\n🔧 SOLUZIONI SUGGERITE:")
        print("   1. AUMENTA commitment_cost → 1.0 o 2.0")
        print("   2. USA VQ weight scheduling più aggressivo")
        print("   3. Pre-train l'encoder senza VQ per alcune epoche")
        print("   4. Aggiungi perplexity loss al training")
    else:
        print("\n✅ Nessun problema grave rilevato!")
    
    print("="*70)
    
    return {
        'num_unique_codes': num_unique_codes,
        'avg_vq_loss': avg_vq_loss,
        'avg_perplexity': avg_perplexity,
        'encoder_std_mean': float(np.mean(encoder_std)),
        'issues': issues
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🔬 DIAGNOSI CODEBOOK COLLAPSE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    try:
        # Carica modello
        model, num_embeddings = load_model(CHECKPOINT_PATH, device)
        print(f"✅ Modello caricato: {num_embeddings} codici nel codebook")
        
        # Carica dataset
        print(f"\n📊 Caricando dataset...")
        train_loader, _, _ = create_unified_dataloaders(
            train_session_ids=TRAINING_SESSION_IDS,
            batch_size=BATCH_SIZE,
            test_split=0.2,
            window_size=60,
            stride=50,
            min_neurons=1,
            num_workers=0,
            use_cross_session_test=False
        )
        
        # DIAGNOSI
        results = diagnose_quantization(model, train_loader, device, NUM_BATCHES_TO_TEST)
        
        print("\n✅ DIAGNOSI COMPLETATA!")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())