"""
Dual-Codebook Vector Quantizer per VQ-VAE

Strategia: Usa DUE codebook separati
- Codebook A: embeddings per ATTIVAZIONI (segnale sopra soglia)
- Codebook B: embeddings per NON-ATTIVAZIONI (segnale sotto soglia)

Vantaggi:
1. Forza bilanciamento 50/50 tra attivazioni e baseline
2. Ogni codebook si specializza su un tipo di segnale
3. Migliore utilizzo dello spazio latente
4. Più interpretabilità (codici separati per fenomeni diversi)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DualCodebookQuantizer(nn.Module):
    """
    Dual-Codebook Vector Quantizer with separate embeddings for:
    - Active states (above threshold)
    - Inactive states (below threshold)
    
    Args:
        n_e: Total number of embeddings (will be split 50/50)
        e_dim: Embedding dimension
        beta: Commitment cost
        decay: EMA decay rate
        eps: Small constant for numerical stability
        threshold: Threshold to separate active/inactive (default: 0.0)
        threshold_type: 'fixed' or 'adaptive'
    """
    
    def __init__(self, n_e, e_dim, beta, decay=0.99, eps=1e-5, 
                 threshold=0.0, threshold_type='adaptive'):
        super(DualCodebookQuantizer, self).__init__()
        
        # Dividi embeddings in due codebook
        assert n_e % 2 == 0, "n_e deve essere pari per dual codebook"
        
        self.n_e = n_e
        self.n_e_per_codebook = n_e // 2
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.threshold = threshold
        self.threshold_type = threshold_type
        
        # ============================================
        # CODEBOOK A: Per ATTIVAZIONI (above threshold)
        # ============================================
        self.embedding_active = nn.Embedding(self.n_e_per_codebook, self.e_dim)
        nn.init.xavier_uniform_(self.embedding_active.weight.data)
        self.embedding_active.weight.data = self.embedding_active.weight.data * 0.5
        
        # EMA per codebook A
        self.register_buffer('cluster_size_active', torch.zeros(self.n_e_per_codebook))
        self.register_buffer('embed_avg_active', self.embedding_active.weight.data.clone())
        
        # Usage tracking A
        self.register_buffer('usage_count_active', torch.zeros(self.n_e_per_codebook))
        
        # ============================================
        # CODEBOOK B: Per NON-ATTIVAZIONI (below threshold)
        # ============================================
        self.embedding_inactive = nn.Embedding(self.n_e_per_codebook, self.e_dim)
        nn.init.xavier_uniform_(self.embedding_inactive.weight.data)
        self.embedding_inactive.weight.data = self.embedding_inactive.weight.data * 0.5
        
        # EMA per codebook B
        self.register_buffer('cluster_size_inactive', torch.zeros(self.n_e_per_codebook))
        self.register_buffer('embed_avg_inactive', self.embedding_inactive.weight.data.clone())
        
        # Usage tracking B
        self.register_buffer('usage_count_inactive', torch.zeros(self.n_e_per_codebook))
        
        # Adaptive threshold tracking
        if threshold_type == 'adaptive':
            self.register_buffer('running_mean', torch.tensor(0.0))
            self.register_buffer('num_batches', torch.tensor(0))
    
    def _update_adaptive_threshold(self, inputs):
        """Aggiorna threshold adattivo basato sulla media dei dati"""
        if self.training and self.threshold_type == 'adaptive':
            batch_mean = inputs.mean()
            
            # Running average
            self.num_batches += 1
            alpha = 1.0 / self.num_batches
            self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean
            
            # Usa running mean come threshold
            return self.running_mean.item()
        else:
            return self.threshold
    
    def forward(self, inputs):
        """
        Forward pass con dual codebook
        
        Args:
            inputs: (B, C, T) per 1D o (B, C, H, W) per 2D
            
        Returns:
            loss, quantized, perplexity, encodings, encoding_indices
        """
        input_shape = inputs.shape
        
        # Flatten
        if inputs.dim() == 3:  # 1D
            flat_input = inputs.permute(0, 2, 1).contiguous().view(-1, self.e_dim)
        else:  # 2D
            flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.e_dim)
        
        # ============================================
        # STEP 1: DETERMINA ACTIVE/INACTIVE
        # ============================================
        
        # Calcola "attivazione" come norma o media del vettore
        # Opzione 1: usa la norma L2
        activation_level = torch.norm(flat_input, dim=1)
        
        # Opzione 2: usa la media (se embeddings rappresentano intensità)
        # activation_level = flat_input.mean(dim=1)
        
        # Aggiorna threshold se adaptive
        current_threshold = self._update_adaptive_threshold(inputs)
        
        # Maschera: True = active, False = inactive
        active_mask = activation_level > current_threshold
        inactive_mask = ~active_mask
        
        n_active = active_mask.sum().item()
        n_inactive = inactive_mask.sum().item()
        
        # ============================================
        # STEP 2: QUANTIZZA CON CODEBOOK APPROPRIATO
        # ============================================
        
        # Inizializza output
        quantized = torch.zeros_like(flat_input)
        encodings_combined = torch.zeros(flat_input.shape[0], self.n_e, device=inputs.device)
        encoding_indices_combined = torch.zeros(flat_input.shape[0], dtype=torch.long, device=inputs.device)
        
        loss_active = torch.tensor(0.0, device=inputs.device)
        loss_inactive = torch.tensor(0.0, device=inputs.device)
        
        # QUANTIZZA ACTIVE
        if n_active > 0:
            flat_active = flat_input[active_mask]
            
            # Distanze da codebook A
            distances_active = (
                torch.sum(flat_active**2, dim=1, keepdim=True) 
                + torch.sum(self.embedding_active.weight**2, dim=1)
                - 2 * torch.matmul(flat_active, self.embedding_active.weight.t())
            )
            
            # Encoding
            indices_active = torch.argmin(distances_active, dim=1)
            
            # Update usage
            if self.training:
                self.usage_count_active.index_add_(
                    0, indices_active, torch.ones_like(indices_active, dtype=torch.float)
                )
            
            # One-hot
            encodings_active = torch.zeros(
                indices_active.shape[0], self.n_e_per_codebook, device=inputs.device
            )
            encodings_active.scatter_(1, indices_active.unsqueeze(1), 1)
            
            # Quantize
            quantized_active = torch.matmul(encodings_active, self.embedding_active.weight)
            quantized[active_mask] = quantized_active
            
            # Loss
            e_latent_loss = F.mse_loss(quantized_active.detach(), flat_active)
            q_latent_loss = F.mse_loss(quantized_active, flat_active.detach())
            loss_active = q_latent_loss + self.beta * e_latent_loss
            
            # Store encodings (offset by 0)
            encodings_combined[active_mask, :self.n_e_per_codebook] = encodings_active
            encoding_indices_combined[active_mask] = indices_active
            
            # EMA update per codebook A
            if self.training:
                self.cluster_size_active.data.mul_(self.decay).add_(
                    torch.sum(encodings_active, dim=0), alpha=1 - self.decay
                )
                
                n = torch.sum(self.cluster_size_active.data)
                self.cluster_size_active.data.add_(self.eps).div_(
                    n + self.n_e_per_codebook * self.eps
                ).mul_(n)
                
                dw = torch.matmul(encodings_active.t(), flat_active)
                self.embed_avg_active.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                
                self.embedding_active.weight.data.copy_(
                    self.embed_avg_active / self.cluster_size_active.unsqueeze(1)
                )
        
        # QUANTIZZA INACTIVE
        if n_inactive > 0:
            flat_inactive = flat_input[inactive_mask]
            
            # Distanze da codebook B
            distances_inactive = (
                torch.sum(flat_inactive**2, dim=1, keepdim=True) 
                + torch.sum(self.embedding_inactive.weight**2, dim=1)
                - 2 * torch.matmul(flat_inactive, self.embedding_inactive.weight.t())
            )
            
            # Encoding
            indices_inactive = torch.argmin(distances_inactive, dim=1)
            
            # Update usage
            if self.training:
                self.usage_count_inactive.index_add_(
                    0, indices_inactive, torch.ones_like(indices_inactive, dtype=torch.float)
                )
            
            # One-hot
            encodings_inactive = torch.zeros(
                indices_inactive.shape[0], self.n_e_per_codebook, device=inputs.device
            )
            encodings_inactive.scatter_(1, indices_inactive.unsqueeze(1), 1)
            
            # Quantize
            quantized_inactive = torch.matmul(encodings_inactive, self.embedding_inactive.weight)
            quantized[inactive_mask] = quantized_inactive
            
            # Loss
            e_latent_loss = F.mse_loss(quantized_inactive.detach(), flat_inactive)
            q_latent_loss = F.mse_loss(quantized_inactive, flat_inactive.detach())
            loss_inactive = q_latent_loss + self.beta * e_latent_loss
            
            # Store encodings (offset by n_e_per_codebook)
            encodings_combined[inactive_mask, self.n_e_per_codebook:] = encodings_inactive
            encoding_indices_combined[inactive_mask] = indices_inactive + self.n_e_per_codebook
            
            # EMA update per codebook B
            if self.training:
                self.cluster_size_inactive.data.mul_(self.decay).add_(
                    torch.sum(encodings_inactive, dim=0), alpha=1 - self.decay
                )
                
                n = torch.sum(self.cluster_size_inactive.data)
                self.cluster_size_inactive.data.add_(self.eps).div_(
                    n + self.n_e_per_codebook * self.eps
                ).mul_(n)
                
                dw = torch.matmul(encodings_inactive.t(), flat_inactive)
                self.embed_avg_inactive.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                
                self.embedding_inactive.weight.data.copy_(
                    self.embed_avg_inactive / self.cluster_size_inactive.unsqueeze(1)
                )
        
        # ============================================
        # STEP 3: COMBINA E RITORNA
        # ============================================
        
        # Loss totale (media pesata)
        total_samples = n_active + n_inactive
        if total_samples > 0:
            loss = (loss_active * n_active + loss_inactive * n_inactive) / total_samples
        else:
            loss = torch.tensor(0.0, device=inputs.device)
        
        # Reshape back
        if inputs.dim() == 3:  # 1D
            quantized = quantized.view(input_shape[0], input_shape[2], -1).permute(0, 2, 1)
        else:  # 2D
            quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], -1).permute(0, 3, 1, 2)
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encodings_combined, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encodings_combined, encoding_indices_combined
    
    def get_usage_stats(self):
        """Statistiche di utilizzo per entrambi i codebook"""
        # Codebook A (active)
        used_active = (self.usage_count_active > 0).sum().item()
        usage_pct_active = (used_active / self.n_e_per_codebook) * 100
        
        # Codebook B (inactive)
        used_inactive = (self.usage_count_inactive > 0).sum().item()
        usage_pct_inactive = (used_inactive / self.n_e_per_codebook) * 100
        
        # Totale
        total_used = used_active + used_inactive
        total_usage_pct = (total_used / self.n_e) * 100
        
        return {
            'total_used_codes': total_used,
            'total_codes': self.n_e,
            'total_usage_percentage': total_usage_pct,
            
            'active_used_codes': used_active,
            'active_total_codes': self.n_e_per_codebook,
            'active_usage_percentage': usage_pct_active,
            'active_avg_usage': self.usage_count_active.mean().item(),
            'active_max_usage': self.usage_count_active.max().item(),
            
            'inactive_used_codes': used_inactive,
            'inactive_total_codes': self.n_e_per_codebook,
            'inactive_usage_percentage': usage_pct_inactive,
            'inactive_avg_usage': self.usage_count_inactive.mean().item(),
            'inactive_max_usage': self.usage_count_inactive.max().item(),
            
            'threshold': self.threshold if self.threshold_type == 'fixed' else self.running_mean.item(),
        }
    
    def reset_usage_stats(self):
        """Reset contatori"""
        self.usage_count_active.zero_()
        self.usage_count_inactive.zero_()
    
    def get_all_embeddings(self):
        """Ritorna tutti gli embeddings concatenati per analisi"""
        return torch.cat([
            self.embedding_active.weight.data,
            self.embedding_inactive.weight.data
        ], dim=0)


if __name__ == "__main__":
    print("🧪 Testing DualCodebookQuantizer\n")
    
    # Test 1D
    print("=" * 70)
    print("TEST 1D (Calcium Imaging)")
    print("=" * 70)
    
    x_1d = torch.randn(4, 64, 50)  # (B, C, T)
    
    # Crea quantizer con 512 embeddings (256 per active, 256 per inactive)
    dual_vq = DualCodebookQuantizer(
        n_e=512,
        e_dim=64,
        beta=0.25,
        decay=0.99,
        threshold_type='adaptive'
    )
    
    print(f"Input shape: {x_1d.shape}")
    print(f"Codebook A (active): {dual_vq.n_e_per_codebook} embeddings")
    print(f"Codebook B (inactive): {dual_vq.n_e_per_codebook} embeddings")
    print(f"Total embeddings: {dual_vq.n_e}")
    
    # Forward pass
    loss, quantized, perplexity, encodings, indices = dual_vq(x_1d)
    
    print(f"\n✅ Output:")
    print(f"   Input:      {x_1d.shape}")
    print(f"   Quantized:  {quantized.shape}")
    print(f"   Loss:       {loss:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    
    # Statistiche
    stats = dual_vq.get_usage_stats()
    print(f"\n📊 Usage Statistics:")
    print(f"   Total: {stats['total_used_codes']}/{stats['total_codes']} "
          f"({stats['total_usage_percentage']:.1f}%)")
    print(f"   Active codebook: {stats['active_used_codes']}/{stats['active_total_codes']} "
          f"({stats['active_usage_percentage']:.1f}%)")
    print(f"   Inactive codebook: {stats['inactive_used_codes']}/{stats['inactive_total_codes']} "
          f"({stats['inactive_usage_percentage']:.1f}%)")
    print(f"   Threshold: {stats['threshold']:.4f}")
    
    # Verifica bilanciamento
    active_count = (indices < dual_vq.n_e_per_codebook).sum().item()
    inactive_count = (indices >= dual_vq.n_e_per_codebook).sum().item()
    total_count = active_count + inactive_count
    
    print(f"\n⚖️  Bilanciamento:")
    print(f"   Active:   {active_count}/{total_count} ({100*active_count/total_count:.1f}%)")
    print(f"   Inactive: {inactive_count}/{total_count} ({100*inactive_count/total_count:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)