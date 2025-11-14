"""
Fixed Vector Quantizer for VQ-VAE
- Consistent 5-value returns
- Better initialization
- EMA updates
- Usage tracking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# ImprovedVectorQuantizer
# -----------------------------------------------------------------------------
# Versione migliorata del Vector Quantizer per VQ-VAE.
# - Inizializzazione controllata con Xavier e scaling ridotto.
# - Aggiornamento delle embedding via EMA (Exponential Moving Average).
# - Supporto sia per input 1D (B,C,T) che 2D (B,C,H,W).
# - Restituisce sempre 5 valori: (loss, quantized, perplexity, encodings, indices).
# - Tiene traccia dell’utilizzo dei codici nel codebook.
# Include anche la classe VectorQuantizer originale per compatibilità retroattiva.
# ----------------------------------------------------------------------------- 

class ImprovedVectorQuantizer(nn.Module):
    """
    Improved VectorQuantizer with better initialization and EMA updates.
    Handles both 1D and 2D data.
    
    FIXED: Returns consistent 5 values matching training expectations
    """
    
    def __init__(self, n_e, e_dim, beta, decay=0.99, eps=1e-5):
        super(ImprovedVectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        
        # Embeddings
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # ✅ Inizializzazione più controllata
        nn.init.xavier_uniform_(self.embedding.weight.data)
        # Scale down per evitare valori troppo grandi
        self.embedding.weight.data = self.embedding.weight.data * 0.5
        
        # EMA parameters
        self.register_buffer('cluster_size', torch.zeros(n_e))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        
        # Usage tracking
        self.register_buffer('usage_count', torch.zeros(n_e))
    
    def forward(self, inputs):
        """
        Handle both 1D (B,C,T) and 2D (B,C,H,W) inputs
        
        Returns: (loss, quantized, perplexity, encodings, encoding_indices)
        """
        input_shape = inputs.shape
        
        # Flatten to (batch*spatial, channels)
        if inputs.dim() == 3:  # 1D case
            flat_input = inputs.permute(0, 2, 1).contiguous().view(-1, self.e_dim)
        else:  # 2D case
            flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.e_dim)
        
        # Calculate distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding - find closest embedding
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Update usage tracking
        if self.training:
            self.usage_count.index_add_(0, encoding_indices, 
                                       torch.ones_like(encoding_indices, dtype=torch.float))
        
        # One-hot encodings
        encodings = torch.zeros(encoding_indices.shape[0], self.n_e, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight)
        
        # Reshape back
        if inputs.dim() == 3:  # 1D case
            quantized = quantized.view(input_shape[0], input_shape[2], -1).permute(0, 2, 1)
        else:  # 2D case
            quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], -1).permute(0, 3, 1, 2)
        
        # Loss calculation
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.beta * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # 🔥 EMA update (only in training)
        if self.training:
            # Update cluster sizes
            self.cluster_size.data.mul_(self.decay).add_(
                torch.sum(encodings, dim=0), alpha=1 - self.decay)
            
            # Laplace smoothing
            n = torch.sum(self.cluster_size.data)
            self.cluster_size.data.add_(self.eps).div_(n + self.n_e * self.eps).mul_(n)
            
            # Update embeddings
            dw = torch.matmul(encodings.t(), flat_input)
            self.embed_avg.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # Normalize embeddings
            self.embedding.weight.data.copy_(
                self.embed_avg / self.cluster_size.unsqueeze(1)
            )
        
        # ✅ Return 5 values consistently
        return loss, quantized, perplexity, encodings, encoding_indices
    
    def get_usage_stats(self):
        """Get codebook usage statistics"""
        used_codes = (self.usage_count > 0).sum().item()
        usage_pct = (used_codes / self.n_e) * 100
        
        return {
            'used_codes': used_codes,
            'total_codes': self.n_e,
            'usage_percentage': usage_pct,
            'avg_usage': self.usage_count.mean().item(),
            'max_usage': self.usage_count.max().item(),
        }
    
    def reset_usage_stats(self):
        """Reset usage counter"""
        self.usage_count.zero_()

        # old VectorQuantizer 
class VectorQuantizer(nn.Module):
    """
    Original VectorQuantizer - kept for backward compatibility
    """
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0, 1.0)

    def forward(self, z):
        if z.dim() == 3:
            return self._forward_1d(z)
        elif z.dim() == 4:
            return self._forward_2d(z)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {z.dim()}D")
    
    def _forward_1d(self, z):
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 2, 1).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
    
    def _forward_2d(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


if __name__ == "__main__":
    print("Testing ImprovedVectorQuantizer:")
    
    # Test 1D
    x_1d = torch.randn(2, 64, 50)
    vq = ImprovedVectorQuantizer(512, 64, 0.25)
    loss, quantized, perplexity, encodings, indices = vq(x_1d)
    print(f"1D: Input {x_1d.shape}, Output {quantized.shape}, Perplexity {perplexity:.2f}")
    print(f"    Returns 5 values: ✅")
    
    # Test 2D
    x_2d = torch.randn(2, 64, 8, 8)
    loss, quantized, perplexity, encodings, indices = vq(x_2d)
    print(f"2D: Input {x_2d.shape}, Output {quantized.shape}, Perplexity {perplexity:.2f}")
    print(f"    Returns 5 values: ✅")
    
    # Test usage stats
    stats = vq.get_usage_stats()
    print(f"\nUsage stats: {stats['usage_percentage']:.1f}% codebook used")