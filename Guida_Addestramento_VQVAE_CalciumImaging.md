# Guida Completa: Addestramento VQ-VAE per Calcium Imaging

## Indice
1. [Panoramica del Sistema](#panoramica-del-sistema)
2. [Architettura del Modello](#architettura-del-modello)
3. [Processo di Addestramento](#processo-di-addestramento)
4. [Componenti Dettagliati](#componenti-dettagliati)
5. [Loss Functions e Ottimizzazione](#loss-functions-e-ottimizzazione)
6. [Salvataggio e Valutazione](#salvataggio-e-valutazione)

---

## 1. Panoramica del Sistema

Il codice implementa un **Vector Quantized Variational AutoEncoder (VQ-VAE)** ottimizzato per l'analisi di dati di calcium imaging da neuroni del cervello di topi. L'obiettivo è scoprire "building blocks" universali dell'attività neurale attraverso la quantizzazione vettoriale.

### Flusso Generale del Sistema

```
Dati Neurali (30 neuroni × 60 timesteps)
           ↓
    [ENCODER] - Compressione
           ↓
    Latent Space (128 canali × 15 timesteps)
           ↓
    [QUANTIZER] - Discretizzazione in codebook
           ↓
    Quantized Latent (128 canali × 15 timesteps)
           ↓
    [DECODER] - Ricostruzione
           ↓
    Ricostruito (30 neuroni × 60 timesteps)
```

---

## 2. Architettura del Modello

### 2.1 Configurazione del Modello (dal codice)

Per il caso di **singolo neurone** (file: `train_di_1_neurone.py`):

```python
MODEL_CONFIG = {
    'num_neurons': 1,              # Numero di neuroni in input
    'num_hiddens': 128,            # Dimensione hidden layer
    'num_residual_layers': 3,      # Strati residuali
    'num_residual_hiddens': 64,    # Hidden nei blocchi residuali
    'num_embeddings': 2048,         # Dimensione massima del codebook (16,32,64,128,256,512,1024)
    'embedding_dim': 128,          # Dimensionalità embedding (32,32,32,64,64,128,128)
    'commitment_cost': 0.5,        # Peso della commitment loss
    'dropout_rate': 0.0,           # Dropout per regolarizzazione
    'use_quantizer': True          # Abilita quantizzazione
}
```

### 2.2 Struttura Completa del Modello

Il modello `CalciumVQVAE` è composto da tre componenti principali:

```python
class CalciumVQVAE(nn.Module):
    def __init__(self, ...):
        # 1. ENCODER
        self.encoder = CalciumEncoder(...)
        
        # 2. PRE-QUANTIZATION (proiezione)
        self.pre_quantization_conv = nn.Conv1d(
            in_channels=num_hiddens,     
            out_channels=embedding_dim,   
            kernel_size=1
        )
        
        # 3. QUANTIZER
        self.vector_quantization = ImprovedVectorQuantizer(
            n_e=num_embeddings,          # 2048 codici
            e_dim=embedding_dim,         # 128 dimensioni
            beta=commitment_cost         # 0.5
        )
        
        # 4. DECODER
        self.decoder = CalciumDecoder(...)
```

---

## 3. Processo di Addestramento

### 3.1 Preparazione dei Dati

Il sistema carica dati da **sessioni multiple** dell'Allen Brain Observatory:

```python
# File: datasets/calcium.py
TRAINING_SESSION_IDS = [
    650389887,  # Sessione 1 - VISp
    642883713,  # Sessione 2 - VISp
    # ... 10 sessioni totali
]

# Caricamento unificato
train_loader, test_loader, dataset_info = create_unified_dataloaders(
    train_session_ids=TRAINING_SESSION_IDS,
    batch_size=64,
    window_size=60,    # 60 timesteps (2 secondi @ 30Hz)
    stride=50,         # Overlap di 10 timesteps
    min_neurons=1      # 1 neurone per finestra
)
```

**Preprocessing dei dati**:
1. Estrazione da file HDF5/NWB
2. Clipping outlier: `[-2, 5]` (valori ΔF/F)
3. Sliding windows temporali: 60 timesteps con stride 50
4. Shuffle dei batch per il training

### 3.2 Loop di Addestramento Principale

```python
def train_epoch_single_neuron(model, dataloader, optimizer, device, epoch, config):
    model.train()
    
    # === 1. VQ WEIGHT SCHEDULING ===
    # Warmup graduale: prima ricostruzione, poi quantizzazione
    warmup_epochs = 30
    if epoch < warmup_epochs:
        vq_weight = 0.001 * (epoch / warmup_epochs)
        phase = "WARMUP"
    else:
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        vq_weight = 0.001 + 0.1 * progress  # Max 0.101
        phase = "VQ_ACTIVE"
    
    for neural_data in dataloader:
        neural_data = neural_data.to(device)
        
        # === 2. FORWARD PASS ===
        vq_loss, recon, perplexity, _, _ = model(neural_data)
        
        # === 3. LOSS COMPUTATION ===
        # a) Reconstruction Loss (MSE + Temporal Smoothness)
        combined_loss, recon_loss, smooth_loss = \
            compute_loss_with_temporal_smoothness(recon, neural_data, alpha=0.05)
        
        # b) Total Loss con VQ scheduling
        loss = combined_loss + vq_weight * vq_loss
        
        # === 4. BACKWARD PASS ===
        optimizer.zero_grad()
        loss.backward()
        
        # === 5. GRADIENT CLIPPING ===
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # === 6. OPTIMIZATION STEP ===
        optimizer.step()
    
    return metrics
```

### 3.3 Scheduling e Regolarizzazione

#### VQ Weight Scheduling

Il peso della VQ loss aumenta gradualmente per stabilizzare il training:

```
Epoch    VQ Weight    Fase
------------------------------------
0-30     0.000-0.001  WARMUP (focus: ricostruzione)
31-100   0.001-0.05   TRANSIZIONE
100+     0.05-0.101   VQ_ACTIVE (quantizzazione piena)
```

#### Learning Rate Scheduling

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=max_epochs,
    eta_min=1e-6  # LR minimo
)

# Annealing cosinusoidale:
# LR iniziale: 0.0005
# LR finale:   0.000001
```

---

## 4. Componenti Dettagliati

### 4.1 ENCODER: Compressione Neurale

```python
class CalciumEncoder(nn.Module):
    def __init__(self, in_channels=1, num_hiddens=128, ...):
        # === DOWNSAMPLING PROGRESSIVO ===
        
        # Layer 1: Cattura pattern locali
        self._conv_1 = nn.Conv1d(
            in_channels=1,           # 1 neurone
            out_channels=32,         # num_hiddens//4
            kernel_size=7,           # Kernel largo (±3 timesteps)
            stride=1,                # No downsampling ancora
            padding=3
        )
        
        # Layer 2: Riduzione dimensionalità
        self._conv_2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,         # num_hiddens//2
            kernel_size=5,
            stride=2,                # 60 → 30 timesteps
            padding=2
        )
        
        # Layer 3: Compressione finale
        self._conv_3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,        # num_hiddens
            kernel_size=3,
            stride=2,                # 30 → 15 timesteps
            padding=1
        )
        
        # === RESIDUAL STACK ===
        # Raffinamento senza cambio dimensioni
        self._residual_stack = nn.ModuleList([
            ImprovedResidualBlock(128, 128, 64)
            for _ in range(3)  # num_residual_layers
        ])
        
        # === DROPOUT per regolarizzazione ===
        self._dropout = nn.Dropout(0.0)  # Disabilitato in questo caso
```

**Trasformazioni dimensionali nell'Encoder**:

```
Input:  (batch, 1, 60)      # 1 neurone, 60 timesteps
   ↓ conv_1 (k=7, s=1)
       (batch, 32, 60)      # 32 canali, tempo invariato
   ↓ dropout
   ↓ conv_2 (k=5, s=2)
       (batch, 64, 30)      # 64 canali, tempo dimezzato
   ↓ dropout
   ↓ conv_3 (k=3, s=2)
       (batch, 128, 15)     # 128 canali, tempo ri-dimezzato
   ↓ residual_stack (×3)
Output: (batch, 128, 15)    # Latent compresso
```

#### Blocchi Residuali

```python
class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_channels=128, num_hiddens=128, num_residual_hiddens=64):
        self._block = nn.Sequential(
            nn.GroupNorm(8, 128),           # Normalizzazione
            nn.ReLU(True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=1)
        )
    
    def forward(self, x):
        return x + self._block(x)  # Skip connection
```

**Vantaggi dei blocchi residuali**:
- **Skip connections**: gradiente fluisce meglio
- **GroupNorm**: stabilizza training con batch piccoli
- **Preservano dimensioni**: (128, 15) → (128, 15)

---

### 4.2 QUANTIZER: Discretizzazione nel Codebook

Il quantizzatore trasforma rappresentazioni continue in rappresentazioni discrete.

```python
class ImprovedVectorQuantizer(nn.Module):
    def __init__(self, n_e=2048, e_dim=128, beta=0.5):
        self.n_e = n_e              # Numero di codici (2048)
        self.e_dim = e_dim          # Dimensionalità embedding (128)
        self.beta = beta            # Commitment cost (0.5)
        
        # === CODEBOOK ===
        # Embeddings appresi: 2048 vettori di dimensione 128
        self.embedding = nn.Embedding(2048, 128)
        
        # Inizializzazione Xavier con scaling
        nn.init.xavier_uniform_(self.embedding.weight.data)
        self.embedding.weight.data *= 0.5
        
        # === EMA (Exponential Moving Average) ===
        # Buffers per aggiornamenti EMA del codebook
        self.register_buffer('cluster_size', torch.zeros(2048))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        
        # === USAGE TRACKING ===
        # Conta quante volte ogni codice è usato
        self.register_buffer('usage_count', torch.zeros(2048))
```

#### Forward Pass del Quantizer

```python
def forward(self, inputs):
    # Input: (batch, 128, 15)
    
    # === 1. FLATTEN ===
    # (batch, 128, 15) → (batch, 15, 128) → (batch*15, 128)
    flat_input = inputs.permute(0, 2, 1).contiguous().view(-1, 128)
    
    # === 2. COMPUTE DISTANCES ===
    # Distanza L2: ||z - e||² = ||z||² + ||e||² - 2*z·e
    distances = (
        torch.sum(flat_input**2, dim=1, keepdim=True) +      # ||z||²
        torch.sum(self.embedding.weight**2, dim=1) -         # ||e||²
        2 * torch.matmul(flat_input, self.embedding.weight.t())  # -2*z·e
    )
    # Shape: (batch*15, 2048)
    
    # === 3. FIND NEAREST EMBEDDING ===
    encoding_indices = torch.argmin(distances, dim=1)
    # Shape: (batch*15,) - indice del codice più vicino per ogni posizione
    
    # === 4. ONE-HOT ENCODING ===
    encodings = torch.zeros(flat_input.shape[0], 2048, device=inputs.device)
    encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
    # Shape: (batch*15, 2048) - one-hot per codice selezionato
    
    # === 5. QUANTIZE ===
    # Sostituisci z con e (embedding dal codebook)
    quantized = torch.matmul(encodings, self.embedding.weight)
    # Shape: (batch*15, 128)
    
    # === 6. RESHAPE BACK ===
    quantized = quantized.view(batch, 15, 128).permute(0, 2, 1)
    # Shape: (batch, 128, 15)
    
    # === 7. COMPUTE LOSSES ===
    # a) Encoder loss: ||sg[z] - e||²
    e_latent_loss = F.mse_loss(quantized.detach(), inputs)
    
    # b) Commitment loss: ||z - sg[e]||²
    q_latent_loss = F.mse_loss(quantized, inputs.detach())
    
    # c) Total VQ loss
    loss = q_latent_loss + beta * e_latent_loss
    
    # === 8. STRAIGHT-THROUGH ESTIMATOR ===
    # Gradiente passa attraverso come se fosse identità
    quantized = inputs + (quantized - inputs).detach()
    
    # === 9. PERPLEXITY (codebook usage) ===
    avg_probs = torch.mean(encodings, dim=0)  # (2048,)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    
    # === 10. EMA UPDATE (solo in training) ===
    if self.training:
        # Update cluster sizes
        self.cluster_size.data.mul_(0.99).add_(
            torch.sum(encodings, dim=0), alpha=0.01
        )
        
        # Update embeddings
        dw = torch.matmul(encodings.t(), flat_input)
        self.embed_avg.data.mul_(0.99).add_(dw, alpha=0.01)
        
        # Normalize
        self.embedding.weight.data.copy_(
            self.embed_avg / self.cluster_size.unsqueeze(1)
        )
        
        # Track usage
        self.usage_count.index_add_(0, encoding_indices, 
                                   torch.ones_like(encoding_indices, dtype=torch.float))
    
    return loss, quantized, perplexity, encodings, encoding_indices
```

#### Perché il Quantizer è Importante

1. **Discretizzazione**: Forza la rete a rappresentare i dati con un vocabolario finito di 2048 "parole neurali"
2. **Compressione**: Ogni posizione temporale (15 in totale) è rappresentata da un indice 0-2047 invece di 128 float
3. **Interpretabilità**: I codici del codebook possono essere analizzati come pattern neurali universali
4. **Regolarizzazione**: Limita la capacità espressiva prevenendo overfitting

#### Metriche del Quantizer

**Perplexity**:
```
Perplexity = exp(-Σ p(e) * log(p(e)))

Interpretazione:
- Perplexity = 1:     Solo 1 codice usato (codebook collapse!)
- Perplexity = 2048:  Tutti i codici usati uniformemente (ideale)
- Perplexity = 200:   ~10% del codebook attivo (realistico)
```

**Usage Statistics**:
```python
stats = model.get_codebook_usage()
# {
#   'used_codes': 487,           # Codici effettivamente usati
#   'total_codes': 2048,
#   'usage_percentage': 23.8,    # Percentuale di utilizzo
#   'avg_usage': 12.5,           # Utilizzo medio per codice
#   'max_usage': 1543            # Codice più utilizzato
# }
```

---

### 4.3 DECODER: Ricostruzione Neurale

Il decoder inverte il processo dell'encoder, espandendo da latent a spazio originale.

```python
class CalciumDecoder(nn.Module):
    def __init__(self, embedding_dim=128, num_hiddens=128, 
                 output_channels=1, ...):
        
        # === RESIDUAL STACK (raffinamento iniziale) ===
        self._residual_stack = nn.ModuleList([
            ImprovedResidualBlock(128, 128, 64)
            for _ in range(3)
        ])
        
        # === UPSAMPLING PROGRESSIVO ===
        
        # Layer 1: Upsampling + aumento canali
        self._conv_transpose_1 = nn.ConvTranspose1d(
            in_channels=128,
            out_channels=128,        # num_hiddens
            kernel_size=3,
            stride=2,                # 15 → 30 timesteps
            padding=1,
            output_padding=1
        )
        
        # Layer 2: Ulteriore upsampling + riduzione canali
        self._conv_transpose_2 = nn.ConvTranspose1d(
            in_channels=128,
            out_channels=64,         # num_hiddens//2
            kernel_size=5,
            stride=2,                # 30 → 60 timesteps
            padding=2,
            output_padding=0
        )
        
        # Layer 3: Proiezione finale su spazio neurale
        self._conv_final = nn.Conv1d(
            in_channels=64,
            out_channels=1,          # output_channels (1 neurone)
            kernel_size=7,
            stride=1,
            padding=3
        )
        
        self._dropout = nn.Dropout(0.0)
```

**Trasformazioni dimensionali nel Decoder**:

```
Input:  (batch, 128, 15)        # Quantized latent
   ↓ residual_stack (×3)
       (batch, 128, 15)         # Raffinamento
   ↓ conv_transpose_1 (k=3, s=2)
       (batch, 128, 30)         # Tempo raddoppiato
   ↓ dropout
   ↓ conv_transpose_2 (k=5, s=2)
       (batch, 64, 60)          # Tempo ri-raddoppiato
   ↓ dropout
   ↓ conv_final (k=7, s=1)
Output: (batch, 1, 60)          # Ricostruzione neurale
```

#### Simmetria Encoder-Decoder

```
ENCODER                    DECODER
--------                   --------
1 neuron, 60 timesteps →   ← 1 neuron, 60 timesteps
   ↓ (k=7, s=1)               (k=7, s=1) ↑
32 channels, 60 →              ← 64 channels, 60
   ↓ (k=5, s=2)               (k=5, s=2, transpose) ↑
64 channels, 30 →              ← 128 channels, 30
   ↓ (k=3, s=2)               (k=3, s=2, transpose) ↑
128 channels, 15               128 channels, 15
   ↓ residual                 residual ↑
128 channels, 15 → [QUANTIZER] → 128 channels, 15
```

---

## 5. Loss Functions e Ottimizzazione

### 5.1 Loss Totale

La loss totale è una combinazione ponderata di diverse componenti:

```python
def compute_loss_with_temporal_smoothness(recon, target, alpha=0.05):
    # === 1. RECONSTRUCTION LOSS ===
    # MSE tra ricostruzione e target
    recon_loss = F.mse_loss(recon, target)
    
    # === 2. TEMPORAL SMOOTHNESS LOSS ===
    # Penalizza differenze temporali brusche
    diff_recon = recon[:, :, 1:] - recon[:, :, :-1]
    diff_target = target[:, :, 1:] - target[:, :, :-1]
    smooth_loss = F.mse_loss(diff_recon, diff_target)
    
    # === 3. COMBINED ===
    total_loss = recon_loss + alpha * smooth_loss
    
    return total_loss, recon_loss, smooth_loss

# Nel training loop:
# Loss finale = combined_loss + vq_weight * vq_loss
```

#### Componenti della Loss

| Loss | Formula | Peso | Scopo |
|------|---------|------|-------|
| **Reconstruction** | MSE(x_recon, x) | 1.0 | Fedeltà della ricostruzione |
| **Temporal Smoothness** | MSE(Δx_recon, Δx) | 0.05 | Coerenza temporale |
| **VQ Loss** | ‖sg[z]-e‖² + β‖z-sg[e]‖² | 0.001-0.101 | Quantizzazione efficace |

**Nota**: `sg[]` indica stop-gradient (gradiente bloccato).

### 5.2 VQ Loss Dettagliata

```python
# Nel quantizer:
# e_latent_loss: spinge encoder a produrre embeddings vicini al codebook
e_latent_loss = F.mse_loss(quantized.detach(), inputs)

# q_latent_loss: spinge codebook ad avvicinarsi agli embeddings dell'encoder
q_latent_loss = F.mse_loss(quantized, inputs.detach())

# Commitment loss: impedisce all'encoder di "scappare" dal codebook
vq_loss = q_latent_loss + beta * e_latent_loss
#           ↑                ↑
#      Update codebook   Commit encoder
```

**Interpretazione geometrica**:
- `q_latent_loss`: il codebook "insegue" le rappresentazioni dell'encoder
- `β * e_latent_loss`: l'encoder è "vincolato" a rimanere vicino al codebook
- Il bilanciamento (β=0.5) previene "codebook collapse"

### 5.3 Ottimizzatore e Hyperparameters

```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0.0005,              # Learning rate iniziale
    weight_decay=0.0001,    # L2 regularization
    betas=(0.9, 0.999)      # Momentum per primo e secondo momento
)

# Cosine annealing per decadimento LR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=205,              # Totale epoche
    eta_min=1e-6            # LR minimo
)
```

**Evoluzione del Learning Rate**:
```
Epoch    Learning Rate
------------------------
0        0.0005000
50       0.0003827
100      0.0001913
150      0.0000382
200      0.0000010
```

---

## 6. Salvataggio e Valutazione

### 6.1 Checkpoint del Modello

Quando il modello migliora (based on test correlation), viene salvato:

```python
if current_test_corr > best_test_corr:
    best_test_corr = current_test_corr
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': MODEL_CONFIG,
        'best_correlation': best_test_corr,
        'test_metrics': test_metrics
    }
    
    torch.save(checkpoint, './results/best_single_neuron_model_2048_pt2.pth')
```

**Contenuto del file .pth**:
- `model_state_dict`: Tutti i pesi del modello (encoder, quantizer, decoder)
- `optimizer_state_dict`: Stato dell'ottimizzatore (per resume training)
- `model_config`: Hyperparameters per ricreare l'architettura
- `best_correlation`: Metrica di performance
- `test_metrics`: Dizionario con MSE, R², SNR, etc.

### 6.2 Metriche di Valutazione

```python
def evaluate_model_single_neuron(model, dataloader, device):
    model.eval()
    
    with torch.no_grad():
        for neural_data in dataloader:
            vq_loss, recon, perplexity, _, _ = model(neural_data)
            # Accumula predizioni
    
    # Calcola metriche
    metrics = {
        'test/mse': mean_squared_error(...),
        'test/mae': mean_absolute_error(...),
        'test/r_squared': r2_score(...),
        'test/correlation_mean': np.mean(correlations),
        'test/snr_db': 10 * log10(signal/noise),
        'test/perplexity': perplexity_avg
    }
    
    return metrics
```

#### Metriche Chiave

| Metrica | Formula | Interpretazione | Target |
|---------|---------|-----------------|--------|
| **MSE** | E[(x - x̂)²] | Errore quadratico medio | < 0.01 |
| **MAE** | E[‖x - x̂‖] | Errore assoluto medio | < 0.1 |
| **R²** | 1 - SS_res/SS_tot | Varianza spiegata | > 0.8 |
| **Correlation** | ρ(x, x̂) | Correlazione temporale | > 0.9 |
| **SNR** | 10log₁₀(P_signal/P_noise) | Rapporto segnale-rumore | > 20 dB |
| **Perplexity** | exp(-Σp log p) | Utilizzo codebook | 100-500 |

### 6.3 Visualizzazioni

Il sistema genera visualizzazioni dei top sample (quelli con ampiezza massima):

```python
def visualize_single_neuron_reconstruction(original, reconstructed, epoch):
    # === 1. SELEZIONE SAMPLE ===
    # Calcola ampiezza per ogni sample
    amplitudes = [max(trace) - min(trace) for trace in original]
    
    # Top 4 sample con ampiezza maggiore
    top_indices = argsort(amplitudes)[-4:]
    
    # === 2. PLOT PER OGNI SAMPLE ===
    for sample_idx in top_indices:
        # Plot 1: Overlay originale vs ricostruzione
        # Plot 2: Errore temporale (focus sui picchi)
        # Plot 3: Zoom sui picchi più alti
    
    # Salva su W&B
    wandb.log({"reconstructions": wandb.Image(fig)})
```

**Elementi visualizzati**:
- Traccia originale vs ricostruzione
- Errore assoluto nel tempo
- Zoom sui picchi neurali (eventi importanti)
- Statistiche: correlazione, ampiezza, numero picchi

---

## 7. Pipeline Completa di Addestramento

### 7.1 Sequenza Esecutiva

```
1. INIZIALIZZAZIONE
   ├─ Carica configurazione modello
   ├─ Inizializza W&B per logging
   └─ Setup device (CUDA/CPU)

2. CARICAMENTO DATI
   ├─ Estrai dati da Allen Brain Observatory
   ├─ Preprocessing (clipping, normalizzazione)
   ├─ Crea sliding windows (60 timesteps)
   └─ Split train/test (80%/20%)

3. CREAZIONE MODELLO
   ├─ Inizializza CalciumVQVAE
   ├─ Carica su GPU
   └─ Setup optimizer + scheduler

4. TRAINING LOOP (200+ epoche)
   ├─ Per ogni epoch:
   │  ├─ WARMUP (0-30): Focus ricostruzione
   │  ├─ VQ_ACTIVE (30+): Attiva quantizzazione
   │  ├─ Forward pass su batch
   │  ├─ Calcola loss totale
   │  ├─ Backward + gradient clipping
   │  └─ Update pesi
   │
   ├─ Ogni 5 epoche:
   │  ├─ Valuta su test set
   │  ├─ Log metriche su W&B
   │  └─ Salva checkpoint se migliorato
   │
   └─ Ogni 20 epoche:
      └─ Genera visualizzazioni

5. EARLY STOPPING
   └─ Stop se no improvement per 40 epoche

6. SALVATAGGIO FINALE
   ├─ Salva best model checkpoint
   ├─ Log metriche finali
   └─ Chiudi W&B run
```

### 7.2 Esempio di Output Training

```
🧠 SINGLE NEURON VQ-VAE TRAINING
==================================================================
✅ Model config:
   num_neurons: 1
   num_hiddens: 128
   num_embeddings: 2048
   embedding_dim: 128
   commitment_cost: 0.5
==================================================================
💻 Using device: cuda

📊 Loading datasets...
   Sessions: 10
   Train samples: 16384
   Test samples: 4096

🏗️ Creating model...
✅ Model created: 1,234,567 parameters

🚀 Starting training...

Epoch   0 [WARMUP]:
  Train Loss: 0.245631
  VQ Weight: 0.000033
  Perplexity: 12.4

Epoch   5 [WARMUP]:
  Test MSE: 0.123456
  Test Corr: 0.6234 ± 0.1823
  R²: 0.5123, SNR: 18.2 dB
  Perplexity: 45.2

Epoch  30 [VQ_ACTIVE]:
  Test MSE: 0.045123
  Test Corr: 0.8912 ± 0.0456
  R²: 0.8523, SNR: 28.7 dB
  Perplexity: 187.3
  💾 Saved best model (corr=0.8912)

...

Epoch 150:
  Test MSE: 0.012345
  Test Corr: 0.9456 ± 0.0234
  R²: 0.9234, SNR: 35.2 dB
  Perplexity: 423.1
  💾 Saved best model (corr=0.9456)

🎯 FINAL EVALUATION:
Test MSE: 0.012345
Test Correlation: 0.9456
R²: 0.9234
SNR: 35.2 dB

✅ Training completed!
```

---

## 8. Riassunto dei Concetti Chiave

### 8.1 Perché VQ-VAE per Calcium Imaging?

1. **Rappresentazione Discreta**: I neuroni hanno stati discreti (spike/no-spike), VQ-VAE cattura questa natura discreta
2. **Codebook Interpretabile**: Ogni codice può rappresentare un "pattern neurale" ricorrente
3. **Compressione Efficace**: 60 timesteps × 1 neurone → 15 indici (2048 possibili) = compressione ~30x
4. **Generalizzazione Cross-Session**: Pattern appresi da 10 sessioni possono applicarsi a nuove sessioni

### 8.2 Differenze da Autoencoder Classico

| Aspetto | Autoencoder | VQ-VAE |
|---------|-------------|--------|
| Latent Space | Continuo | Discreto (codebook) |
| Rappresentazione | Vettori densi | Indici + embeddings |
| Loss | Solo ricostruzione | Ricostruzione + VQ + commitment |
| Interpretabilità | Difficile | Codici analizzabili |
| Compressione | Lossy continua | Lossy + quantizzazione |

### 8.3 Problemi Comuni e Soluzioni

#### Problema 1: Codebook Collapse
**Sintomo**: Perplexity < 10, solo pochi codici usati

**Soluzioni implementate**:
- EMA updates per stabilizzare embeddings
- Commitment cost (β=0.5) per bilanciare encoder/codebook
- VQ weight scheduling (warmup graduale)
- Inizializzazione Xavier scalata (×0.5)

#### Problema 2: Mode Collapse
**Sintomo**: Ricostruzione identica per tutti gli input

**Soluzioni implementate**:
- Dropout (se abilitato) per regolarizzazione
- Temporal smoothness loss per coerenza
- Gradient clipping (max_norm=1.0)
- Data augmentation via multi-session training

#### Problema 3: Overfitting
**Sintomo**: Train loss << Test loss

**Soluzioni implementate**:
- Weight decay (0.0001) in AdamW
- Test set da sessioni diverse (cross-session)
- Early stopping (patience=40)
- Learning rate annealing

---

## 9. Come Utilizzare il Modello Addestrato

### 9.1 Caricare il Checkpoint

```python
# Carica checkpoint
checkpoint = torch.load('./results/best_single_neuron_model_2048_pt2.pth')

# Ricrea modello con configurazione salvata
model = CalciumVQVAE(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Best correlation: {checkpoint['best_correlation']:.4f}")
```

### 9.2 Inferenza su Nuovi Dati

```python
# Prepara dati
neural_trace = torch.randn(1, 1, 60).to(device)  # (batch, neurons, time)

with torch.no_grad():
    # Forward pass completo
    vq_loss, recon, perplexity, quantized, encodings = model(neural_trace)
    
    # Oppure step-by-step
    encoded = model.encode(neural_trace)      # (1, 128, 15)
    decoded = model.decode(encoded)           # (1, 1, 60)

# Estrai codici usati
_, _, _, _, encoding_indices = model.vector_quantization(encoded)
print(f"Codici usati: {encoding_indices.cpu().numpy()}")
# Output: [234, 1023, 567, 12, ...]  (15 indici)
```

### 9.3 Analisi del Codebook

```python
# Ottieni statistiche utilizzo
stats = model.get_codebook_usage()
print(f"Codici attivi: {stats['used_codes']}/{stats['total_codes']}")
print(f"Utilizzo: {stats['usage_percentage']:.1f}%")

# Estrai embeddings del codebook
codebook_weights = model.vector_quantization.embedding.weight.data
print(f"Codebook shape: {codebook_weights.shape}")  # (2048, 128)

# Analizza codice specifico
code_idx = 234
embedding_vector = codebook_weights[code_idx].cpu().numpy()
print(f"Codice {code_idx}: {embedding_vector[:5]}...")  # Primi 5 valori

# Decodifica codice per vedere pattern neurale
with torch.no_grad():
    # Crea latent con solo questo codice (replicato)
    test_latent = embedding_vector.repeat(15, 1).unsqueeze(0).to(device)
    test_latent = test_latent.permute(0, 2, 1)  # (1, 128, 15)
    
    neural_pattern = model.decode(test_latent)  # (1, 1, 60)
    
import matplotlib.pyplot as plt
plt.plot(neural_pattern[0, 0].cpu().numpy())
plt.title(f"Pattern Neurale del Codice {code_idx}")
plt.show()
```

---

## 10. Prossimi Passi e Miglioramenti

### 10.1 Esperimenti Suggeriti

1. **Multi-Neuron Extension**: Aumentare `num_neurons` da 1 a 30 per catturare correlazioni tra neuroni
2. **Hierarchical VQ-VAE**: Aggiungere livelli multipli di quantizzazione per pattern gerarchici
3. **Conditional Generation**: Condizionare la generazione su stimoli visivi (dai metadati Allen)
4. **Codebook Pruning**: Eliminare codici inutilizzati e ri-addestrare
5. **Transfer Learning**: Pre-addestrare su VISp, fine-tune su VISl/VISam

### 10.2 Possibili Applicazioni

- **Neural Decoding**: Predire stimoli visivi da attività neurale quantizzata
- **Anomaly Detection**: Identificare pattern neurali atipici
- **Data Compression**: Comprimere grandi dataset di calcium imaging
- **Generative Modeling**: Campionare nuove traiettorie neurali plausibili

---

## Conclusione

Questo documento ha coperto:
- **Architettura completa** del modello VQ-VAE per calcium imaging
- **Funzionamento dettagliato** di encoder, quantizer, decoder
- **Pipeline di addestramento** con scheduling e regolarizzazione
- **Loss functions** e loro interpretazione
- **Metriche di valutazione** e visualizzazioni
- **Utilizzo pratico** del modello addestrato

Il sistema implementa un approccio sofisticato per apprendere rappresentazioni discrete dell'attività neurale, bilanciando compressione, interpretabilità e qualità della ricostruzione attraverso tecniche avanzate come VQ weight scheduling, EMA updates, e temporal smoothness regularization.

---

**File di riferimento**:
- `models/vqvae.py` - Architettura principale
- `models/encoder.py` - Encoder 1D per calcium imaging
- `models/quantizer.py` - Quantizzatore con EMA
- `datasets/calcium.py` - Caricamento dati Allen
- `scripts/train_di_1_neurone.py` - Training loop completo

**Autore**: Claudio  
**Data**: Novembre 2025  
**Versione**: 1.0
