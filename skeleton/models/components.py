from typing import List, Tuple, Union

import torch
import torch.nn as nn
from models.stgcn import ST_GCN_layer


class Encoder(nn.Module):
    def __init__(self, layer_channels:List[int], n_frames:int, graph_size:List[int]=[3, 17, 17], edge=True) -> None:
        """
        Class that implements a Space-Time-Separable Graph Convolutional Encoder (STS-GCN).

        Args:
            input_dim (int): number of coordinates of the input
            layer_channels (List[int]): list of channel dimension for each layer
            hidden_dimension (int): dimension of the hidden layer
            n_frames (int): number of frames of the input pose sequence
            n_joints (int): number of joints of the input pose sequence
            dropout (float): dropout probability
            bias (bool, optional): whether to use bias in the convolutional layers. Defaults to True.
        """
        
        super().__init__()
        
        # Set the model's parameters
        self.input_dim = layer_channels[0]
        self.layer_channels = layer_channels[1:]
        self.n_frames = n_frames
        self.graph_size = graph_size
        self.edge = edge
        
        # Build the model
        self.build_model()
        

    def build_model(self):
        """
        Build the model.

        Returns:
            nn.ModuleList: list of the model's layers
        """
        
        input_channels = self.input_dim
        kernel_size = [self.n_frames//2 + 1, self.graph_size[0]]
        stride = 1
        model_layers = nn.ModuleList()
        for channels in self.layer_channels:
            model_layers.append(
                ST_GCN_layer(in_channels=input_channels, 
                                     out_channels=channels,
                                     kernel_size=kernel_size,
                                     stride=stride))
            input_channels = channels
        self.model_layers = model_layers

        if self.edge:
            self.edge_importance = nn.ParameterList([
                    nn.Parameter(torch.ones(self.graph_size))
                    for _ in self.model_layers
            ])

        else:
            self.edge_importance = [1 for _ in self.model_layers]
        
        
    def forward(self, X:torch.Tensor, A) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, input_channels, n_frames, n_joints]
            t (torch.Tensor): time tensor of shape [batch_size, n_frames]. Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape [batch_size, hidden_dimension, n_frames, n_joints]
            List[torch.Tensor]: list of the output tensors of each intermediate layer
        """
        for layer, importance in zip(self.model_layers, self.edge_importance):
            X = layer(X, A*importance)
        
        return X
    
    
 
    
class Decoder(nn.Module):
    def __init__(self, layer_channels:List[int], n_frames:int, graph_size:List[int]=[3, 17, 17], edge=True) -> None:
        """
        Class that implements a Space-Time-Separable Graph Convolutional Decoder (STS-GCN).

        Args:
            output_dim (int): number of coordinates of the output
            layer_channels (List[int]): list of channel dimension for each layer (in the same order as the encoder's layers)
            hidden_dimension (int): dimension of the hidden layer
            n_frames (int): number of frames of the input pose sequence
            n_joints (int): number of joints of the input pose sequence
            dropout (float): dropout probability
            bias (bool, optional): whether to use bias in the convolutional layers. Defaults to True.
        """
        
        super().__init__()
        
        # Set the model's parameters
        self.layer_channels = layer_channels[:-1][::-1]
        self.hidden_dimension = layer_channels[-1]
        self.n_frames = n_frames
        self.edge = edge
        self.graph_size = graph_size
        
        # Build the model
        self.build_model()
        
    
    def build_model(self):
        """
        Build the model.
        """
        
        input_channels = self.hidden_dimension
        kernel_size = [self.n_frames//2 + 1, self.graph_size[0]]
        stride = 1
        model_layers = nn.ModuleList()
        for channels in self.layer_channels:
            model_layers.append(
                ST_GCN_layer(in_channels=input_channels, 
                                     out_channels=channels,
                                     kernel_size=kernel_size,
                                     stride=stride))
            input_channels = channels
        
        self.model_layers = model_layers
        if self.edge:
            self.edge_importance = nn.ParameterList([
                    nn.Parameter(torch.ones(self.graph_size))
                    for _ in self.model_layers
            ])
        else:
            self.edge_importance = [1 for _ in self.model_layers]

    def forward(self, X:torch.Tensor, A) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, hidden_dimension, n_frames, n_joints]
            t (torch.Tensor): time tensor of shape [batch_size, n_frames]. Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape [batch_size, output_dim, n_frames, n_joints]
        """
        for layer, importance in zip(self.model_layers, self.edge_importance):
            X = layer(X, A*importance)
        
        return X
            