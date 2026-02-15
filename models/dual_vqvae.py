"""
VQ-VAE con Dual Codebook per Calcium Imaging

Usa DualCodebookQuantizer invece del quantizer standard
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import CalciumEncoder
from models.vqvae import CalciumDecoder
from models.dual_quantizer import DualCodebookQuantizer


class DualCalciumVQVAE(nn.Module):
    """
    VQ-VAE con DUAL CODEBOOK per calcium imaging.
    
    - Codebook A: embeddings per attivazioni
    - Codebook B: embeddings per non-attivazioni
    
    Args:
        num_neurons: Numero di neuroni (input channels)
        num_hiddens: Hidden dimension
        num_residual_layers: Numero di layer residuali
        num_residual_hiddens: Hidden dimension nei residual blocks
        num_embeddings: TOTALE embeddings (verrà diviso 50/50)
        embedding_dim: Dimensione embedding
        commitment_cost: Beta per commitment loss
        dropout_rate: Dropout rate
        use_quantizer: Se False, non quantizza (perfect reconstruction)
        threshold: Soglia per separare active/inactive
        threshold_type: 'fixed' o 'adaptive'
    """
    
    def __init__(self, num_neurons=30, num_hiddens=128, num_residual_layers=2, 
                 num_residual_hiddens=32, num_embeddings=512, embedding_dim=64, 
                 commitment_cost=0.25, dropout_rate=0.3, use_quantizer=True,
                 threshold=0.0, threshold_type='adaptive'):
        super(DualCalciumVQVAE, self).__init__()
        
        self.use_quantizer = use_quantizer
        
        # Encoder
        self.encoder = CalciumEncoder(
            num_neurons,           
            num_hiddens,           
            num_residual_layers,   
            num_residual_hiddens,
            dropout_rate=dropout_rate
        )
        
        # Pre-quantization conv
        self.pre_quantization_conv = nn.Conv1d(
            in_channels=num_hiddens, 
            out_channels=embedding_dim,
            kernel_size=1, 
            stride=1
        )
        
        # 🔥 DUAL CODEBOOK QUANTIZER
        self.vector_quantization = DualCodebookQuantizer(
            n_e=num_embeddings,      # Verrà diviso in due codebook da num_embeddings/2
            e_dim=embedding_dim, 
            beta=commitment_cost,
            decay=0.99,
            eps=1e-5,
            threshold=threshold,
            threshold_type=threshold_type
        )
        
        # Decoder
        self.decoder = CalciumDecoder(
            embedding_dim,         
            num_hiddens,           
            num_residual_layers,   
            num_residual_hiddens,  
            num_neurons,           
            dropout_rate           
        )
        
        print(f"✅ DualCalciumVQVAE creato:")
        print(f"   Total embeddings: {num_embeddings}")
        print(f"   Active codebook: {num_embeddings//2}")
        print(f"   Inactive codebook: {num_embeddings//2}")
        print(f"   Threshold type: {threshold_type}")

    def forward(self, x):
        """
        Forward pass
        
        Returns:
            vq_loss, x_recon, perplexity, quantized, encodings
        """
        # Encode
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        
        # Normalize
        z = F.layer_norm(z, [z.size(1), z.size(2)])
        z = torch.clamp(z, min=-10.0, max=10.0)
        
        if self.use_quantizer:
            # Dual codebook quantization con VINCOLO
            # Passa x_original per forzare selezione codebook basata su attività reale
            vq_loss, quantized, perplexity, encodings, encoding_indices = \
                self.vector_quantization(z, x_original=x)
        else:
            # No quantization
            quantized = z
            vq_loss = torch.tensor(0.0, device=z.device)
            perplexity = torch.tensor(float(self.vector_quantization.n_e), device=z.device)
            encodings = None
            encoding_indices = None
        
        # Decode
        x_recon = self.decoder(quantized)
        
        # Match dimensioni
        if x_recon.shape[2] != x.shape[2]:
            x_recon = F.interpolate(
                x_recon, size=x.shape[2], mode='linear', align_corners=False
            )
        
        return vq_loss, x_recon, perplexity, quantized, encodings
    
    def encode(self, x):
        """Encode input"""
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        
        if self.use_quantizer:
            _, quantized, _, _, _ = self.vector_quantization(z)
        else:
            quantized = z
            
        return quantized
    
    def decode(self, quantized):
        """Decode quantized representation"""
        return self.decoder(quantized)
    
    def get_codebook_usage(self):
        """Statistiche di utilizzo dei codebook"""
        if hasattr(self.vector_quantization, 'get_usage_stats'):
            return self.vector_quantization.get_usage_stats()
        else:
            return {'total_usage_percentage': 100.0}
    
    def get_all_embeddings(self):
        """Ritorna tutti gli embeddings per analisi"""
        if hasattr(self.vector_quantization, 'get_all_embeddings'):
            return self.vector_quantization.get_all_embeddings()
        else:
            return self.vector_quantization.embedding.weight.data


if __name__ == "__main__":
    print("🧠 Testing DualCalciumVQVAE\n")
    
    # Crea modello
    model = DualCalciumVQVAE(
        num_neurons=30, 
        num_hiddens=128, 
        num_residual_layers=2,
        num_residual_hiddens=32, 
        num_embeddings=512,      # 256 active + 256 inactive
        embedding_dim=64,
        commitment_cost=0.25,
        threshold_type='adaptive'
    )
    
    # Test input
    x = torch.randn(4, 30, 60)
    print(f"📊 Input shape: {x.shape}\n")
    
    # Forward pass
    vq_loss, x_recon, perplexity, quantized, encodings = model(x)
    
    print(f"✅ Output shapes:")
    print(f"   Reconstruction: {x_recon.shape}")
    print(f"   Quantized: {quantized.shape}")
    print(f"   VQ Loss: {vq_loss:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    
    # Verifica shape
    assert x_recon.shape == x.shape
    print(f"\n✅ Shape match: {x_recon.shape} == {x.shape}")
    
    # Reconstruction quality
    recon_mse = F.mse_loss(x_recon, x)
    print(f"🎯 Reconstruction MSE: {recon_mse:.6f}")
    
    # Codebook usage
    usage_stats = model.get_codebook_usage()
    print(f"\n📚 Codebook usage:")
    print(f"   Total: {usage_stats['total_usage_percentage']:.1f}%")
    print(f"   Active codebook: {usage_stats['active_usage_percentage']:.1f}%")
    print(f"   Inactive codebook: {usage_stats['inactive_usage_percentage']:.1f}%")
    print(f"   Threshold: {usage_stats['threshold']:.4f}")
    
    # Test encode/decode
    encoded = model.encode(x)
    decoded = model.decode(encoded)
    print(f"\n🔄 Encode->Decode shape: {decoded.shape}")
    
    print(f"\n✨ Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"✅ All tests passed!")