import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.encoder import CalciumEncoder
from models.quantizer import VectorQuantizer


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
        
        # Progressive upsampling (fixed output_padding)
        self._conv_transpose_1 = nn.ConvTranspose1d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        
        # ✅ FIXED: output_padding deve essere < stride (1 < 2)
        self._conv_transpose_2 = nn.ConvTranspose1d(
            in_channels=num_hiddens,
            out_channels=num_hiddens//2,
            kernel_size=5, stride=2, padding=2, output_padding=1  # ← Era 3, ora 1
        )
        
        self._conv_final = nn.Conv1d(
            in_channels=num_hiddens//2,
            out_channels=output_channels,
            kernel_size=7, stride=1, padding=3
        )
        
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Apply residual blocks
        for block in self._residual_stack:
            x = block(x)
        
        # Upsample
        x = F.relu(self._conv_transpose_1(x))
        x = self._dropout(x)
        x = F.relu(self._conv_transpose_2(x))
        x = self._dropout(x)
        x = self._conv_final(x)
        
        return x


class CalciumVQVAE(nn.Module):
    """
    Simplified VQ-VAE optimized for calcium imaging data.
    
    Features:
    - 1D convolutions for temporal neural data
    - ONLY Standard VectorQuantizer
    - NO behavior prediction
    - Focus on reconstruction quality
    """
    
    def __init__(self, num_neurons=30, num_hiddens=128, num_residual_layers=2, 
                 num_residual_hiddens=32, num_embeddings=512, embedding_dim=64, 
                 commitment_cost=0.25, dropout_rate=0.3):
        super(CalciumVQVAE, self).__init__()
        
        # Encoder
        self.encoder = CalciumEncoder(
            num_neurons, num_hiddens, num_residual_layers, 
            num_residual_hiddens, dropout_rate
        )
        
        # Pre-quantization convolution
        self.pre_quantization_conv = nn.Conv1d(
            in_channels=num_hiddens, 
            out_channels=embedding_dim,
            kernel_size=1, stride=1
        )
        
        # ✅ SOLO Standard Vector Quantizer
        self.vector_quantization = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )
        
        # Decoder
        self.decoder = CalciumDecoder(
            embedding_dim, num_hiddens, num_residual_layers,
            num_residual_hiddens, num_neurons, dropout_rate
        )

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (B, num_neurons, time_steps)
            
        Returns:
            tuple: (vq_loss, x_recon, perplexity, quantized, encodings)
        """
        # Encode
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        
        # Quantize  
        vq_loss, quantized, perplexity, encodings, encoding_indices = self.vector_quantization(z)
        
        # Decode
        x_recon = self.decoder(quantized)
        
        # ✅ Assicura che le dimensioni finali combacino
        if x_recon.shape[2] != x.shape[2]:
            x_recon = F.interpolate(
                x_recon, size=x.shape[2], mode='linear', align_corners=False
            )

        return vq_loss, x_recon, perplexity, quantized, encodings
    
    def encode(self, x):
        """Encode input to quantized representation."""
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        _, quantized, _, _, _ = self.vector_quantization(z)
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


def create_simple_calcium_vqvae(config=None):
    """
    Factory function to create CalciumVQVAE.
    
    Args:
        config: dict with model parameters
        
    Returns:
        CalciumVQVAE model
    """
    default_config = {
        'num_neurons': 30,
        'num_hiddens': 128,
        'num_residual_layers': 2,
        'num_residual_hiddens': 32,
        'num_embeddings': 512,
        'embedding_dim': 64,
        'commitment_cost': 0.25,
        'dropout_rate': 0.3
    }
    
    # Update with user config
    if config:
        default_config.update(config)
    
    return CalciumVQVAE(**default_config)


if __name__ == "__main__":
    print("🧠 Testing CalciumVQVAE:")
    
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
    
    # Test input (batch=4, neurons=30, timesteps=50)
    x = torch.randn(4, 30, 50)
    print(f"📊 Input shape: {x.shape}")
    
    # Forward pass
    vq_loss, x_recon, perplexity, quantized, encodings = model(x)
    
    print(f"✅ Output shapes:")
    print(f"   Reconstruction: {x_recon.shape}")
    print(f"   Quantized: {quantized.shape}")
    print(f"   VQ Loss: {vq_loss:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    
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