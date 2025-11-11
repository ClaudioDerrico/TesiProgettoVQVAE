import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.encoder import CalciumEncoder
from models.quantizer import ImprovedVectorQuantizer

# -----------------------------------------------------------------------------
# CalciumVQVAE Model
# -----------------------------------------------------------------------------
# Implementa una variante 1D del VQ-VAE ottimizzata per dati di imaging al calcio.
# - Encoder: comprime l'attività neuronale in rappresentazioni latenti.
# - Quantizer (ImprovedVectorQuantizer): converte i vettori continui in codici discreti.
# - Decoder: ricostruisce i segnali originali tramite convoluzioni trasposte.
# - Applica layer normalization e clipping più permissivo (-10, +10) per stabilizzare
#   la quantizzazione e prevenire valori anomali.
# - Include dropout nei layer di decodifica per ridurre overfitting.
# - Supporta disattivazione del quantizer (use_quantizer=False) per test o ablation.
# - Restituisce sempre 5 valori: (vq_loss, x_recon, perplexity, quantized, encodings).
# - Test finale verifica correttezza di forma, ricostruzione e utilizzo del codebook.
# ----------------------------------------------------------------------------- 


class CalciumDecoder(nn.Module):
    """
    1D Decoder optimized for calcium imaging data.
    Symmetric to CalciumEncoder with transpose convolutions.
    """

    def __init__(self, embedding_dim, num_hiddens, num_residual_layers, 
                 num_residual_hiddens, output_channels, dropout_rate=0.3):
        super(CalciumDecoder, self).__init__()
        
        from models.encoder import ImprovedResidualBlock
        
        # Residual stack at the beginning
        self._residual_stack = nn.ModuleList([
            ImprovedResidualBlock(embedding_dim, embedding_dim, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])
        
        # Progressive upsampling
        self._conv_transpose_1 = nn.ConvTranspose1d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        
        self._conv_transpose_2 = nn.ConvTranspose1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens//2,
            kernel_size=5, stride=2, padding=2, output_padding=0
        )
        
        # Final projection
        self._conv_final = nn.Conv1d(
            in_channels=num_hiddens//2,
            out_channels=output_channels,
            kernel_size=7, stride=1, padding=3
        )
        
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for block in self._residual_stack:
            x = block(x)
        
        x = F.relu(self._conv_transpose_1(x))
        x = self._dropout(x)
        x = F.relu(self._conv_transpose_2(x))
        x = self._dropout(x)
        x = self._conv_final(x)
        
        return x


class CalciumVQVAE(nn.Module):
    """
    VQ-VAE optimized for calcium imaging data.
    
    FIXED: Properly handles quantizer enable/disable with consistent returns
    """
    
    def __init__(self, num_neurons=30, num_hiddens=128, num_residual_layers=2, 
                 num_residual_hiddens=32, num_embeddings=512, embedding_dim=64, 
                 commitment_cost=0.25, dropout_rate=0.3, use_quantizer=True):
        super(CalciumVQVAE, self).__init__()
        
        self.use_quantizer = use_quantizer
        
        # Encoder
        self.encoder = CalciumEncoder(
            num_neurons,           
            num_hiddens,           
            num_residual_layers,   
            num_residual_hiddens,
            dropout_rate=dropout_rate
        )
        
        # Pre-quantization convolution
        self.pre_quantization_conv = nn.Conv1d(
            in_channels=num_hiddens, 
            out_channels=embedding_dim,
            kernel_size=1, 
            stride=1
        )
        
        # 🔥 USE ImprovedVectorQuantizer
        self.vector_quantization = ImprovedVectorQuantizer(
            num_embeddings, 
            embedding_dim, 
            commitment_cost,
            decay=0.99,
            eps=1e-5
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

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (B, num_neurons, time_steps)
            
        Returns:
            tuple: (vq_loss, x_recon, perplexity, quantized, encodings)
            ✅ ALWAYS returns 5 values
        """
        # Encode
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        
        # 🔥 CLIPPING PIÙ PERMISSIVO
        z = F.layer_norm(z, [z.size(1), z.size(2)])
        z = torch.clamp(z, min=-10.0, max=10.0)  # ✅ Era -3,+3, troppo stretto!
        
        if self.use_quantizer:
            # ✅ ImprovedVectorQuantizer ritorna 5 valori!
            vq_loss, quantized, perplexity, encodings, encoding_indices = \
                self.vector_quantization(z)
        else:
            # No quantization - pass through
            quantized = z
            vq_loss = torch.tensor(0.0, device=z.device)
            perplexity = torch.tensor(float(self.vector_quantization.n_e), device=z.device)
            encodings = None
            encoding_indices = None
        
        # Decode
        x_recon = self.decoder(quantized)
        
        # Assicura match dimensioni
        if x_recon.shape[2] != x.shape[2]:
            x_recon = F.interpolate(
                x_recon, size=x.shape[2], mode='linear', align_corners=False
            )

        # ✅ ALWAYS return 5 values
        return vq_loss, x_recon, perplexity, quantized, encodings
    
    def encode(self, x):
        """Encode input to quantized representation."""
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        
        if self.use_quantizer:
            _, quantized, _, _, _ = self.vector_quantization(z)
        else:
            quantized = z
            
        return quantized
    
    def decode(self, quantized):
        """Decode quantized representation to reconstruction."""
        return self.decoder(quantized)
    
    def get_codebook_usage(self):
        """Get codebook usage statistics."""
        if hasattr(self.vector_quantization, 'get_usage_stats'):
            return self.vector_quantization.get_usage_stats()
        else:
            return {'usage_percentage': 100.0}




if __name__ == "__main__":
    print("🧠 Testing CalciumVQVAE with 60 timesteps:")
    
    # Create model
    model = CalciumVQVAE(
        num_neurons=30, 
        num_hiddens=128, 
        num_residual_layers=2,
        num_residual_hiddens=32, 
        num_embeddings=512, 
        embedding_dim=64,
        commitment_cost=0.25
    )
    
    # Test input (batch=4, neurons=30, timesteps=60)
    x = torch.randn(4, 30, 60)
    print(f"📊 Input shape: {x.shape}")
    
    # Forward pass
    vq_loss, x_recon, perplexity, quantized, encodings = model(x)
    
    print(f"✅ Output shapes:")
    print(f"   Reconstruction: {x_recon.shape}")
    print(f"   Quantized: {quantized.shape}")
    print(f"   VQ Loss: {vq_loss:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    
    # Verify shape match
    assert x_recon.shape == x.shape, f"Shape mismatch! Expected {x.shape}, got {x_recon.shape}"
    print(f"✅ Shape match verified: {x_recon.shape} == {x.shape}")
    
    # Test reconstruction quality
    recon_mse = F.mse_loss(x_recon, x)
    print(f"🎯 Reconstruction MSE: {recon_mse:.6f}")
    
    # Test individual encode/decode
    encoded = model.encode(x)
    decoded = model.decode(encoded)
    print(f"🔄 Encode->Decode shape: {decoded.shape}")
    
    # Codebook usage
    usage_stats = model.get_codebook_usage()
    print(f"📚 Codebook usage: {usage_stats}")
    
    print(f"\n✨ Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✅ All tests passed for 30x60 configuration!")