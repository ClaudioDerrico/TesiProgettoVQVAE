"""
Behavioral VQ-VAE: Transfer Learning from Neural Signals to Behavior

Strategy: Use pretrained encoder + codebook (frozen) with new velocity decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BehavioralVQVAE(nn.Module):
    """
    VQ-VAE for VELOCITY prediction using pretrained neural encoder
    
    Architecture:
    - Input: (num_neurons, 1, 60) - all neurons from session
    - Encoder: processes each neuron → (num_neurons, embedding_dim, 15)
    - Aggregate: reshape to (1, num_neurons*embedding_dim, 15)
    - Temporal Pooling: (1, num_neurons*embedding_dim, 1)
    - Decoder: predicts velocity from pooled features
    - Output: 1 velocity value
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
        
        embedding_dim = pretrained_model.vector_quantization.e_dim
        self.compressed_time = 15
        self.embedding_dim = embedding_dim
        
        print(f"\n📊 Velocity Decoder Architecture:")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Compressed time: {self.compressed_time}")
        print(f"   Output: 1 (velocity)")
        print(f"   Strategy: Temporal pooling + compact decoder")
        
        # Temporal pooling to reduce dimensionality
        self.temporal_pool = nn.AdaptiveMaxPool1d(1)
        
        # Decoder will be built dynamically
        self.velocity_decoder = None
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
    
    def _build_decoder(self, num_neurons_times_embedding):
        """Build decoder with LayerNorm for stability"""
        print(f"\n🔨 Building decoder...")
        print(f"   Input after pooling: {num_neurons_times_embedding}")
        
        input_dim = num_neurons_times_embedding
        intermediate_dim = min(1024, max(512, input_dim // 4))
        
        if self.hidden_dim > 0:
            self.velocity_decoder = nn.Sequential(
                nn.Flatten(),
                
                # ✅ Stage 1: Con LayerNorm
                nn.Linear(input_dim, intermediate_dim),
                nn.LayerNorm(intermediate_dim),  # ✅ AGGIUNTO
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                
                # ✅ Stage 2: Con LayerNorm
                nn.Linear(intermediate_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),  # ✅ AGGIUNTO
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                
                # ✅ Stage 3: Con LayerNorm
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),  # ✅ AGGIUNTO
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                
                # Output layer (no norm)
                nn.Linear(self.hidden_dim // 2, 1)
            )
        else:
            self.velocity_decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 1)
            )
        
        # Xavier initialization
        for module in self.velocity_decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        print(f"   ✅ Weights initialized with Xavier + LayerNorm added")
        
        # Move to device
        device = next(self.parameters()).device
        self.velocity_decoder = self.velocity_decoder.to(device)
        
        trainable_params = sum(p.numel() for p in self.velocity_decoder.parameters() if p.requires_grad)
        print(f"✅ Decoder built: {trainable_params:,} trainable params")
        print(f"   Architecture: {input_dim} → {intermediate_dim} → {self.hidden_dim} → {self.hidden_dim//2} → 1 (with LayerNorm)")
        
    def forward(self, x, return_quantized=False):
        """
        Forward pass with temporal pooling
        
        Args:
            x: Neural data (num_neurons, 1, 60)
        
        Returns:
            velocity_output: scalar
        """
        
        num_neurons = x.shape[0]
        
        # ========================================================================
        # ENCODE + QUANTIZE (Frozen but gradients flow)
        # ========================================================================
        
        # NO torch.no_grad() - allow gradient flow!
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        z = F.layer_norm(z, [z.size(1), z.size(2)])
        z = torch.clamp(z, min=-10.0, max=10.0)
        
        vq_loss, quantized, perplexity, encodings, encoding_indices = \
            self.vector_quantization(z)
        
        # quantized shape: (num_neurons, embedding_dim, compressed_time)
        
        # ========================================================================
        # AGGREGATE NEURONS
        # ========================================================================
        
        num_neurons_actual, embedding_dim, compressed_time = quantized.shape
        quantized_aggregated = quantized.reshape(1, num_neurons_actual * embedding_dim, compressed_time)
        
        # quantized_aggregated shape: (1, num_neurons*embedding_dim, compressed_time)
        
        # ========================================================================
        # TEMPORAL POOLING
        # ========================================================================
        
        pooled = self.temporal_pool(quantized_aggregated)
        # pooled shape: (1, num_neurons*embedding_dim, 1)
        
        # ========================================================================
        # BUILD DECODER if needed
        # ========================================================================
        
        if self.velocity_decoder is None:
            num_neurons_times_embedding = num_neurons_actual * embedding_dim
            self._build_decoder(num_neurons_times_embedding)
        
        # ========================================================================
        # VELOCITY PREDICTION
        # ========================================================================
        
        velocity_output = self.velocity_decoder(pooled)
        velocity_output = velocity_output.squeeze()
        
        if return_quantized:
            return velocity_output, quantized_aggregated, vq_loss, perplexity
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
        
        # Aggregate
        num_neurons, embedding_dim, compressed_time = quantized.shape
        quantized_aggregated = quantized.reshape(1, num_neurons * embedding_dim, compressed_time)
        
        return quantized_aggregated
    
    def get_frozen_params_count(self):
        """Count frozen parameters"""
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return frozen, trainable


def load_pretrained_vqvae(checkpoint_path, model_class, model_config, device):
    """Load pretrained VQ-VAE model from checkpoint"""
    print(f"\n🔄 Loading pretrained VQ-VAE from: {checkpoint_path}")
    
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
    """Factory function: Create behavioral model from pretrained checkpoint"""
    
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
    print(f"   Trainable parameters: {trainable:,} (decoder will be added)")
    print(f"   Total parameters: {frozen + trainable:,}")
    
    print("\n✅ Behavioral model created and ready for training!")
    print("   Decoder will be built at first forward pass")
    print("="*70)
    
    return behavioral_model


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing BehavioralVQVAE with ALL neurons\n")
    
    from models.vqvae import CalciumVQVAE
    
    # Create a dummy pretrained model
    print("1. Creating dummy pretrained model...")
    pretrained_model = CalciumVQVAE(
        num_neurons=1,
        num_hiddens=128,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=32,
        embedding_dim=32,
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
    
    # Test forward pass with ALL neurons from a session
    print("3. Testing forward pass with ALL neurons...")
    num_neurons = 142  # Simula sessione con 142 neuroni
    x = torch.randn(num_neurons, 1, 60)  # (num_neurons, 1, 60)
    print(f"   Input shape: {x.shape} (all neurons from session)")
    
    # Forward pass without quantized output
    output = behavioral_model(x, return_quantized=False)
    print(f"   Velocity output shape: {output.shape}")  # Should be scalar
    print(f"   Velocity output: {output.item():.4f}")
    
    # Forward pass with quantized output
    output, quantized, vq_loss, perplexity = behavioral_model(x, return_quantized=True)
    print(f"\n   Velocity output: {output.item():.4f}")
    print(f"   Quantized aggregated: {quantized.shape}")  # (1, 142*32, 15) = (1, 4544, 15)
    print(f"   VQ Loss: {vq_loss:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    
    # Check shapes
    assert output.shape == torch.Size([]), f"Output should be scalar, got {output.shape}"
    assert quantized.shape[0] == 1, "Quantized should have batch=1"
    assert quantized.shape[1] == num_neurons * 32, f"Quantized dim should be {num_neurons*32}"
    print(f"\n✅ All tests passed!")
    
    # Model statistics
    frozen, trainable = behavioral_model.get_frozen_params_count()
    print(f"\n📊 Parameter counts:")
    print(f"   Frozen: {frozen:,}")
    print(f"   Trainable: {trainable:,}")
    print(f"   Total: {frozen + trainable:,}")
    
    # Test with different session size
    print(f"\n4. Testing with different session size...")
    num_neurons2 = 200
    x2 = torch.randn(num_neurons2, 1, 60)
    print(f"   Input shape: {x2.shape}")
    
    # This should fail because decoder is fixed to first session size
    try:
        output2 = behavioral_model(x2, return_quantized=False)
        print(f"   ⚠️ WARNING: Model accepted different session size!")
    except Exception as e:
        print(f"   ✅ Expected: Model is session-specific (frozen to first session size)")
        print(f"   Error: {type(e).__name__}")