import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    VectorQuantizer modified to handle both 1D and 2D data.
    
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    z_e encoder output che coincide con z = self.encoder(x)   
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        For 2D: z.shape = (batch, channel, height, width)
        For 1D: z.shape = (batch, channel, time)
        """
        # Detect if input is 1D or 2D
        if z.dim() == 3:  # 1D case: (batch, channel, time)
            return self._forward_1d(z)
        elif z.dim() == 4:  # 2D case: (batch, channel, height, width)
            return self._forward_2d(z)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {z.dim()}D")
    
    def _forward_1d(self, z):
        """Handle 1D input (batch, channel, time)"""
        # reshape z -> (batch, time, channel) and flatten
        z = z.permute(0, 2, 1).contiguous()  # (batch, time, channel)
        z_flattened = z.view(-1, self.e_dim)  # (batch*time, channel)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape (batch, channel, time)
        z_q = z_q.permute(0, 2, 1).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
    
    def _forward_2d(self, z):
        """Handle 2D input (batch, channel, height, width) - Original implementation"""
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class ImprovedVectorQuantizer(nn.Module):
    """
    Improved VectorQuantizer with better initialization and EMA updates.
    Handles both 1D and 2D data.
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
        self.embedding.weight.data.normal_()
        
        # EMA parameters
        self.register_buffer('cluster_size', torch.zeros(n_e))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        
    def forward(self, inputs):
        """Handle both 1D (B,C,T) and 2D (B,C,H,W) inputs"""
        input_shape = inputs.shape
        
        # Flatten to (batch*spatial, channels)
        if inputs.dim() == 3:  # 1D case
            flat_input = inputs.permute(0, 2, 1).contiguous().view(-1, self.e_dim)
        else:  # 2D case
            flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.e_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.n_e, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight)
        
        # Reshape back
        if inputs.dim() == 3:  # 1D case
            quantized = quantized.view(input_shape[0], input_shape[2], -1).permute(0, 2, 1)
        else:  # 2D case
            quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], -1).permute(0, 3, 1, 2)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.beta * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # EMA update
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                torch.sum(encodings, dim=0), alpha=1 - self.decay)
            
            n = torch.sum(self.cluster_size.data)
            self.cluster_size.data.add_(self.eps).div_(n + self.n_e * self.eps).mul_(n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.embed_avg.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            self.embedding.weight.data.copy_(self.embed_avg / self.cluster_size.unsqueeze(1))
        
        return loss, quantized, perplexity, encodings


if __name__ == "__main__":
    # Test original VQ with both 1D and 2D data
    print("Testing VectorQuantizer:")
    
    # Test 2D (original)
    print("2D test:")
    x_2d = torch.randn(2, 64, 8, 8)
    vq = VectorQuantizer(512, 64, 0.25)
    loss, quantized, perplexity, encodings, indices = vq(x_2d)
    print(f"Input: {x_2d.shape}, Output: {quantized.shape}, Perplexity: {perplexity:.2f}")
    
    # Test 1D (new)
    print("1D test:")
    x_1d = torch.randn(2, 64, 50)  # (batch, channels, time)
    loss, quantized, perplexity, encodings, indices = vq(x_1d)
    print(f"Input: {x_1d.shape}, Output: {quantized.shape}, Perplexity: {perplexity:.2f}")
    
    # Test improved VQ
    print("\nTesting ImprovedVectorQuantizer:")
    vq_improved = ImprovedVectorQuantizer(512, 64, 0.25)
    
    # 1D test
    loss, quantized, perplexity, encodings = vq_improved(x_1d)
    print(f"1D Input: {x_1d.shape}, Output: {quantized.shape}, Perplexity: {perplexity:.2f}")
    
    # 2D test
    loss, quantized, perplexity, encodings = vq_improved(x_2d)
    print(f"2D Input: {x_2d.shape}, Output: {quantized.shape}, Perplexity: {perplexity:.2f}")