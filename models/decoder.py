
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    """nn.ConvTranspose2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1)
    Espande canali da 64 a 128
    NO cambio dimensioni spaziali (stride=1)
    Scopo: Aumentare capacità rappresentativa
    Input: (batch,64,13) → Output: (batch,128,13)
    ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
                128    128    64         3
    Input: (batch, 128,13) → Output: (batch, 128, 13)
    Canali invariati (128 → 128)
    Dimensioni spaziali invariate
    Scopo: Raffinamento features con skip connections

    nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1)
   
    Input: (batch,128,13) → Output: (batch, 64, 26)
    Riduce canali da 128 a 64
    Raddoppia dimensioni spaziali (stride=2)
    nn.ReLU()
    Attivazione ReLU

    nn.ConvTranspose2d(h_dim//2, 3, kernel_size=4, stride=2, padding=1)
    Input: (batch, 64,26) → Output: (1, 30,  50)  


    # Esempio con input (batch=1, in_dim=40, H=10, W=10):

Input:     (1, 64,  13)
    ↓ Layer 1 (kernel=3, stride=1)
           (1, 128, 13)  # Più canali, stesso tempo
    ↓ Layer 2 (ResidualStack)  
           (1, 128, 13)  # Raffinamento, dimensioni invariate
    ↓ Layer 3 (kernel=4, stride=2)
           (1, 64,  26)  # Meno canali, tempo raddoppiato
    ↓ Layer 4 (kernel=4, stride=2)
Output:    (1, 30,  50)  # Neural output, tempo ri-raddoppiato

🔄 Trasformazioni Temporali:

64 → 128 canali: Espansione rappresentativa (13 timesteps invariati)
ResidualStack: Raffinamento qualità (dimensioni invariate)
128 → 64 canali: Riduzione + upsampling temporale (13 → 26)
64 → 30 neuroni: Output finale + upsampling (26 → 50 timesteps originali)

🎯 Obiettivo: Ricostruire 30 tracce neurali su 50 timesteps dalle 64 features quantizzate su 13 timesteps compressi.


In sintesi: Il decoder inverte il processo dell'encoder, espandendo progressivamente da rappresentazione compressa 
a ricostruzione completa

Il kernel è una matrice 4x4 che "scorre" sull'immagine
A ogni passo, guarda 16 pixel contemporaneamente (4×4)
Combina queste informazioni per creare un nuovo pixel

nell'encoder stride riduce dim qua le aumenta

Il kernel decide quanto grande diventa ogni pezzettino quando il computer "ingrandisce" l'immagine.
 Come quando usi la lente d'ingrandimento - più grande è la lente, più grande diventa quello che guardi!
    """


    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
