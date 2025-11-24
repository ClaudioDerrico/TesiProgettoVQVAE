"""
Sequence-to-Sequence Behavioral Regressor

Predice l'INTERA traccia temporale della velocità (60 timesteps)
invece di un singolo valore medio.

Architecture:
- Input: (num_neurons, 60) - attività neurale
- Output: (60,) - velocità per ogni timestep

Questo permette di catturare i picchi istantanei!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceBehavioralRegressor(nn.Module):
    """
    Sequence-to-Sequence regressor per ricostruire traccia temporale completa
    
    Architecture:
    1. Encoder CNN 1D: estrae features temporali da neuroni
    2. Temporal Decoder: ricostruisce sequenza temporale di velocità
    
    Args:
        num_neurons: Numero neuroni (determinato al primo forward)
        hidden_dim: Dimensione hidden layers
        dropout_rate: Dropout rate
        normalize_input: Normalizza input
    """
    
    def __init__(self, num_neurons=None, hidden_dim=256, dropout_rate=0.3, 
                 normalize_input=True, output_length=60):
        super(SequenceBehavioralRegressor, self).__init__()
        
        self.num_neurons = num_neurons
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.normalize_input = normalize_input
        self.output_length = output_length
        
        # Encoder e decoder verranno costruiti al primo forward
        self.encoder = None
        self.decoder = None
        
        print(f"📊 SequenceBehavioralRegressor initialized")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Dropout: {dropout_rate}")
        print(f"   Output length: {output_length} timesteps")
        print(f"   Normalize input: {normalize_input}")
        print(f"   Encoder will be built at first forward")
    
    def _build_model(self, num_neurons, device=None):
        """Costruisce encoder e decoder"""
        print(f"\n🔨 Building Sequence-to-Sequence model for {num_neurons} neurons...")
        
        # ========================================================================
        # ENCODER: Estrae features temporali da attività neurale
        # ========================================================================
        self.encoder = nn.Sequential(
            # Conv1: estrae features locali (60 timesteps → 60)
            nn.Conv1d(num_neurons, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            # Conv2: compressione 2x (60 → 30)
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            # Conv3: compressione 2x (30 → 15)
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        
        # ========================================================================
        # DECODER: Ricostruisce sequenza temporale di velocità
        # ========================================================================
        self.decoder = nn.Sequential(
            # Upsampling 1: 15 → 30 timesteps
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            # Upsampling 2: 30 → 60 timesteps
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            # Refinement layers
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            # Output layer: 64 → 1 channel (velocity)
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
        )
        
        # Xavier initialization
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Move to device
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Model built: {total_params:,} trainable parameters")
        print(f"   Encoder: {num_neurons} neurons → (512, 15) features")
        print(f"   Decoder: (512, 15) → (1, 60) velocity sequence")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch, num_neurons, 60) o (num_neurons, 60)
        
        Returns:
            velocity_sequence: (batch, 60) o (60,)
        """
        
        # Handle single sample
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        
        batch_size, num_neurons, time_steps = x.shape
        
        # Build model if needed
        if self.encoder is None:
            self.num_neurons = num_neurons
            device = x.device
            self._build_model(num_neurons, device=device)
        
        # Verifica consistenza
        if num_neurons != self.num_neurons:
            raise ValueError(
                f"Expected {self.num_neurons} neurons, got {num_neurons}. "
                f"Model is session-specific!"
            )
        
        # Normalizza input
        if self.normalize_input:
            x = F.layer_norm(x, [num_neurons, time_steps])
        
        # Encode: estrai features temporali
        features = self.encoder(x)  # (batch, 512, 15)
        
        # Decode: ricostruisci sequenza velocità
        velocity_seq = self.decoder(features)  # (batch, 1, 60)
        
        # Rimuovi dimensione channel
        velocity_seq = velocity_seq.squeeze(1)  # (batch, 60)
        
        # Match exact output length se necessario
        if velocity_seq.shape[1] != self.output_length:
            velocity_seq = F.interpolate(
                velocity_seq.unsqueeze(1), 
                size=self.output_length, 
                mode='linear', 
                align_corners=False
            ).squeeze(1)
        
        # Squeeze se necessario
        if squeeze_output:
            velocity_seq = velocity_seq.squeeze(0)  # (60,)
        
        return velocity_seq
    
    def get_features(self, x):
        """Estrai features intermedie per analisi"""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        if self.normalize_input:
            x = F.layer_norm(x, [x.size(1), x.size(2)])
        
        features = self.encoder(x)
        return features


if __name__ == "__main__":
    print("🧪 Testing SequenceBehavioralRegressor\n")
    
    model = SequenceBehavioralRegressor(
        hidden_dim=256,
        dropout_rate=0.3,
        normalize_input=True,
        output_length=60
    )
    
    # Test con 181 neuroni
    x = torch.randn(4, 181, 60)
    print(f"📊 Input shape: {x.shape}\n")
    
    output = model(x)
    print(f"\n✅ Output shape: {output.shape}")  # Should be (4, 60)
    
    # Test single sample
    x_single = torch.randn(181, 60)
    output_single = model(x_single)
    print(f"Single sample output shape: {output_single.shape}")  # Should be (60,)
    
    # Verifica dimensioni
    assert output.shape == (4, 60), f"Expected (4, 60), got {output.shape}"
    assert output_single.shape == (60,), f"Expected (60,), got {output_single.shape}"
    
    print(f"\n✨ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ All tests passed!")