import torch.nn as nn
import torch
from typing import List, Tuple, Union

from models.components import Encoder, Decoder


class STAE(nn.Module): 
    
    def __init__(self, layer_channels:List[int]=[2, 64, 128, 128, 256, 256], n_frames:int=12, graph_size:List[int]=[3, 17, 17], edge=True) -> None:
        """
        Space-Time-Separable Autoencoder (STSAE).

        Args:
            c_in (int): number of coordinates of the input
            h_dim (int, optional): dimension of the hidden layer. Defaults to 32.
            latent_dim (int, optional): dimension of the latent space. Defaults to 64.
            n_frames (int, optional): number of frames of the input pose sequence. Defaults to 12.
            n_joints (int, optional): number of joints of the input pose sequence. Defaults to 17.
            layer_channels (List[int], optional): list of channel dimension for each layer. Defaults to [128, 64, 128].
            dropout (float, optional): dropout probability. Defaults to 0.3.
            device (Union[str, torch.DeviceObjType], optional): model device. Defaults to 'cpu'.
        """
        
        super(STAE, self).__init__()
        self.layer_channels = layer_channels
        self.n_frames = n_frames
        self.graph_size = graph_size
        self.edge = edge

        # Build the model
        self.build_model()
        
        
    def build_model(self) -> None:
        """
        Build the model.
        """
        self.encoder = Encoder(layer_channels=self.layer_channels, n_frames=self.n_frames, graph_size=self.graph_size, edge=self.edge)
        
        self.decoder = Decoder(layer_channels=self.layer_channels, n_frames=self.n_frames, graph_size=self.graph_size, edge=self.edge)
        
        
    def forward(self, X:torch.Tensor, A) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            X (torch.Tensor): input pose sequence of shape (batch_size, input_dim, n_frames, n_joints)
            t (torch.Tensor, optional): conditioning signal for the STS-GCN layers. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: reconstructed pose sequence of shape (batch_size, input_dim, n_frames, n_joints)
            and latent representation of the input pose sequence of shape (batch_size, latent_dim)
        """
        Z = self.encoder(X, A)
        X = self.decoder(Z, A)
        return X