"""
Behavioral VQ-VAE: Transfer Learning from Neural Signals to Behavior

Architecture:
- Pretrained Encoder (frozen)
- Pretrained Codebook (frozen) 
- NEW Linear Decoder → 4 behavioral variables

Input: Neural activity (1, neurons, time)
Output: 4 behavioral variables (pupil dilation, vertical position, horizontal position, velocity)

 Modello principale che:

Carica encoder + codebook pre-allenati (frozen)
Sostituisce decoder con linear layer trainable
Predice 4 variabili comportamentali
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BehavioralVQVAE(nn.Module):
    """
    VQ-VAE for VELOCITY prediction using pretrained neural encoder
    
    Key features:
    - Loads pretrained encoder + codebook (frozen)
    - New trainable linear decoder for velocity
    - Output: 1 behavioral variable (velocity)
    
    Args:
        pretrained_model: CalciumVQVAE or DualCalciumVQVAE instance
        freeze_encoder: If True, freeze encoder weights
        freeze_codebook: If True, freeze codebook weights
        hidden_dim: Hidden dimension for velocity decoder (optional)
        dropout_rate: Dropout for regularization
    """
    
    def __init__(self, pretrained_model, freeze_encoder=True, 
                 freeze_codebook=True, hidden_dim=256, dropout_rate=0.3):
        super(BehavioralVQVAE, self).__init__()
        
        # ========================================================================
        # PARTE 1: ENCODER (Pretrained, Frozen)
        # ========================================================================
        self.encoder = pretrained_model.encoder
        self.pre_quantization_conv = pretrained_model.pre_quantization_conv
        
        if freeze_encoder:
            print("🔒 Freezing encoder weights...")
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.pre_quantization_conv.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.pre_quantization_conv.eval()
        
        # ========================================================================
        # PARTE 2: CODEBOOK (Pretrained, Frozen)
        # ========================================================================
        self.vector_quantization = pretrained_model.vector_quantization
        
        if freeze_codebook:
            print("🔒 Freezing codebook weights...")
            for param in self.vector_quantization.parameters():
                param.requires_grad = False
            self.vector_quantization.eval()
        
        # ========================================================================
        # PARTE 3: VELOCITY DECODER (NEW, Trainable)
        # ========================================================================
        
        # Get dimensions from pretrained model
        embedding_dim = pretrained_model.vector_quantization.e_dim
        
        # Compressed time size (depends on encoder architecture)
        # For CalciumEncoder with 2 stride-2 layers: 60 → 30 → 15
        self.compressed_time = 15
        
        # Input size to decoder
        self.flatten_size = embedding_dim * self.compressed_time
        
        print(f"\n📊 Velocity Decoder Architecture:")
        print(f"   Input: ({embedding_dim}, {self.compressed_time}) → Flatten → {self.flatten_size}")
        print(f"   Hidden: {hidden_dim}")
        print(f"   Output: 1 (velocity)")
        
        # Velocity decoder: Flatten → Linear → Output
        if hidden_dim > 0:
            # With hidden layers
            self.velocity_decoder = nn.Sequential(
                nn.Flatten(),  # (B, embedding_dim, compressed_time) → (B, flatten_size)
                
                nn.Linear(self.flatten_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(hidden_dim // 2, 1)  # ✅ Output: 1 (velocity only)
            )
        else:
            # Direct linear projection
            self.velocity_decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.flatten_size, 1)
            )
        
        print(f"✅ Velocity decoder created: {self._count_decoder_params()} trainable params")
    
    def _count_decoder_params(self):
        """Count trainable parameters in velocity decoder"""
        return sum(p.numel() for p in self.velocity_decoder.parameters() if p.requires_grad)
    
    def forward(self, x, return_quantized=False):
        """
        Forward pass
        
        Args:
            x: Neural data (B, neurons, time)
            return_quantized: If True, return quantized representations
        
        Returns:
            If return_quantized=False:
                velocity_output: (B,) - predicted velocity
            If return_quantized=True:
                velocity_output, quantized, vq_loss, perplexity
        """
        
        # ========================================================================
        # ENCODE + QUANTIZE (Frozen)
        # ========================================================================
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Encode
            z = self.encoder(x)
            z = self.pre_quantization_conv(z)
            
            # Normalize
            z = F.layer_norm(z, [z.size(1), z.size(2)])
            z = torch.clamp(z, min=-10.0, max=10.0)
            
            # Quantize
            vq_loss, quantized, perplexity, encodings, encoding_indices = \
                self.vector_quantization(z)
        
        # ========================================================================
        # VELOCITY PREDICTION (Trainable)
        # ========================================================================
        
        # quantized shape: (1, B*embedding_dim, compressed_time) 
        B, embedding_dim, compressed_time = quantized.shape
        quantized=quantized.reshape(1, B*embedding_dim, compressed_time)
        velocity_output = self.velocity_decoder(quantized)
        # velocity_output shape: (B, 1) → squeeze to (B,)
        velocity_output = velocity_output.squeeze(-1)
        
        if return_quantized:
            return velocity_output, quantized, vq_loss, perplexity
        else:
            return velocity_output
    
    def encode(self, x):
        """Encode input to quantized representation (for analysis)"""
        with torch.no_grad():
            z = self.encoder(x)
            z = self.pre_quantization_conv(z)
            z = F.layer_norm(z, [z.size(1), z.size(2)])
            z = torch.clamp(z, min=-10.0, max=10.0)
            _, quantized, _, _, _ = self.vector_quantization(z)
        return quantized
    
    def get_frozen_params_count(self):
        """Count frozen parameters"""
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return frozen, trainable


def load_pretrained_vqvae(checkpoint_path, model_class, model_config, device):
    """
    Load pretrained VQ-VAE model from checkpoint
    
    Args:
        checkpoint_path: Path to .pth file
        model_class: CalciumVQVAE or DualCalciumVQVAE class
        model_config: Dict with model configuration
        device: torch.device
    
    Returns:
        Loaded model instance
    """
    print(f"\n🔄 Loading pretrained VQ-VAE from: {checkpoint_path}")
    
    # Create model
    model = model_class(
        num_neurons=model_config['num_neurons'],
        num_hiddens=model_config['num_hiddens'],
        num_residual_layers=model_config['num_residual_layers'],
        num_residual_hiddens=model_config['num_residual_hiddens'],
        num_embeddings=model_config['num_embeddings'],
        embedding_dim=model_config['embedding_dim'],
        commitment_cost=model_config['commitment_cost'],
        dropout_rate=model_config.get('dropout_rate', 0.0),
        use_quantizer=model_config.get('use_quantizer', True)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model (epoch {checkpoint.get('epoch', 'N/A')})")
        
        if 'best_correlation' in checkpoint:
            print(f"   Best correlation: {checkpoint['best_correlation']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded model")
    
    model = model.to(device)
    model.eval()
    
    return model


def create_behavioral_model_from_checkpoint(
    checkpoint_path, 
    model_class,
    model_config,
    device,
    freeze_encoder=True,
    freeze_codebook=True,
    hidden_dim=256,
    dropout_rate=0.3
):
    """
    Factory function: Create behavioral model from pretrained checkpoint
    
    Args:
        checkpoint_path: Path to pretrained VQ-VAE checkpoint
        model_class: CalciumVQVAE or DualCalciumVQVAE
        model_config: Model configuration dict
        device: torch.device
        freeze_encoder: Freeze encoder weights
        freeze_codebook: Freeze codebook weights
        hidden_dim: Hidden dimension for behavioral decoder
        dropout_rate: Dropout rate
    
    Returns:
        BehavioralVQVAE instance ready for training
    """
    
    print("="*70)
    print("🧠 CREATING BEHAVIORAL VQ-VAE FROM PRETRAINED MODEL")
    print("="*70)
    
    # Load pretrained model
    pretrained_model = load_pretrained_vqvae(
        checkpoint_path, model_class, model_config, device
    )
    
    # Create behavioral model
    behavioral_model = BehavioralVQVAE(
        pretrained_model=pretrained_model,
        freeze_encoder=freeze_encoder,
        freeze_codebook=freeze_codebook,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    )
    
    behavioral_model = behavioral_model.to(device)
    
    # Print summary
    frozen, trainable = behavioral_model.get_frozen_params_count()
    
    print(f"\n📊 Model Summary:")
    print(f"   Frozen parameters: {frozen:,}")
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Total parameters: {frozen + trainable:,}")
    print(f"   Trainable ratio: {100*trainable/(frozen+trainable):.2f}%")
    
    print("\n✅ Behavioral model created and ready for training!")
    print("="*70)
    
    return behavioral_model


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing BehavioralVQVAE\n")
    
    from models.vqvae import CalciumVQVAE
    
    # Create a dummy pretrained model
    print("1. Creating dummy pretrained model...")
    pretrained_model = CalciumVQVAE(
        num_neurons=1,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=512,
        embedding_dim=64,
        commitment_cost=0.25
    )
    print(f"✅ Pretrained model created\n")
    
    # Create behavioral model
    print("2. Creating behavioral model...")
    behavioral_model = BehavioralVQVAE(
        pretrained_model=pretrained_model,
        freeze_encoder=True,
        freeze_codebook=True,
        hidden_dim=256,
        dropout_rate=0.3
    )
    print(f"✅ Behavioral model created\n")
    
    # Test forward pass
    print("3. Testing forward pass...")
    x = torch.randn(4, 1, 60)  # Batch=4, Neurons=1, Time=60
    print(f"   Input shape: {x.shape}")
    
    # Forward pass without quantized output
    output = behavioral_model(x, return_quantized=False)
    print(f"   Velocity output shape: {output.shape}")  # ✅ Changed comment
    
    # Forward pass with quantized output
    output, quantized, vq_loss, perplexity = behavioral_model(x, return_quantized=True)
    print(f"   Velocity output: {output.shape}")
    print(f"   Quantized: {quantized.shape}")
    print(f"   VQ Loss: {vq_loss:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    
    # Check shapes
    assert output.shape == (4,), f"Output shape should be (4,), got {output.shape}"  # ✅ Fixed assertion
    print(f"\n✅ All tests passed!")
    
    # Model statistics
    frozen, trainable = behavioral_model.get_frozen_params_count()
    print(f"\n📊 Parameter counts:")
    print(f"   Frozen: {frozen:,}")
    print(f"   Trainable: {trainable:,}")
    print(f"   Total: {frozen + trainable:,}")