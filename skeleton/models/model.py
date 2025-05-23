from models.stae import STAE
from models.graph import Graph
import torch
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from utils.scoring_utils import get_dataset_scores, smooth_scores, MinMaxNorm
import numpy as np 
from typing import List

class Model(pl.LightningModule):
    """ 
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """
    losses = {'l1':nn.L1Loss, 'smooth_l1':nn.SmoothL1Loss, 'mse':nn.MSELoss}

    def __init__(self, args, metadata, n_joints):
        super(Model, self).__init__()

        self.save_hyperparameters(args, metadata, n_joints)
        self.metadata = metadata
        self.args = args
        self.layer_channels = [args.num_coords] + args.layer_channels
        self.n_frames=args.seg_len
        self.n_joints=n_joints
        self.edge_importance = args.edge_importance
        self.n_cnn_layers=args.n_cnn_layers
        self.dropout = args.dropout
        self.h_dim = args.model_hidden_dim
        self.latent_dim = args.model_latent_dim
        self.num_coords = args.num_coords
        self.learning_rate = args.model_lr
        self.dataset = args.dataset
        self.plot_vis = args.plot_vis
        self.len_plot = args.len_plot * self.n_frames
        self.mask_type = args.mask_type
        self.gamma = args.gamma

        g = Graph(strategy=args.strategy, max_hop=args.max_hops)
        self.A = torch.from_numpy(g.A).float().to(args.device)

        if self.plot_vis:
            check_scene_id = 1
            check_clip_id = 14
            metadata_np = np.array(metadata)
            self.clip_metadata_inds = np.where((metadata_np[:, 1] == check_clip_id) &
                                    (metadata_np[:, 0] == check_scene_id))[0]
        else:
            self.clip_metadata_inds = []

        self.build_model()

    def build_model(self):
        self.model_temporal = STAE(layer_channels=self.layer_channels, n_frames=self.n_frames, graph_size=self.A.size(), edge=self.edge_importance)

    def loss_fn(self, x , y):
        criterion = self.losses[self.args.loss_fn](reduction='mean')
        spa_loss = criterion(x, y)
        temp_loss = criterion((x[:,:,-1] - x[:,:,0]), (y[:,:,-1] - y[:,:,0]))
        loss = self.gamma * temp_loss + (1 - self.gamma) * spa_loss
        return loss
    
    def score_fn(self, x, y):
        spa_score = torch.mean((y-x)**2, axis=(1,2,3))
        temp_score = torch.mean(((y[:,:,-1] - y[:,:,0])-(x[:,:,-1] - x[:,:,0]))**2, axis=(1,2))
        score = self.gamma * temp_score + (1 - self.gamma) * spa_score
        return score

    def forward(self, x):
        _, _, T, _ = x.shape

        x[:,:, :T//2] = 0
        x[:,:, T//2+1:] = 0

        x = self.model_temporal(x, self.A)
            
        return x

    def training_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        loss = self.loss_fn(pred, target)
        self.log('training_loss', loss)
        return loss
    
    def validation_step(self, batch:List[torch.Tensor], batch_idx:int) -> None:
        """
        Validation step of the model. It saves the output of the model and the input data as 
        List[torch.Tensor]: [predicted poses and the loss, tensor_data, transformation_idx, metadata, actual_frames]

        Args:
            batch (List[torch.Tensor]): list containing the following tensors:
                                        - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                        - transformation_idx
                                        - metadata
                                        - actual_frames
            batch_idx (int): index of the batch
        """
        x, target = batch
        pred = self(x)
        score = self.score_fn(pred, target)
        self._validation_output_list.append(score.cpu().detach().numpy().squeeze())
        return


    def on_validation_epoch_start(self) -> None:
        """
        Called when the test epoch begins.
        """
        
        super().on_validation_epoch_start()
        self._validation_output_list = []
        self.plot_images = []
        return


    def on_validation_epoch_end(self) -> float:
        """
        Validation epoch end of the model.

        Returns:
            float: validation auc score
        """
        scores = np.concatenate(self._validation_output_list, axis=0)
        auc_score, _, _, _ =  self.post_processing(scores, True)
        print("\n-------------------------------------------------------")
        print("\033[92m Done with {}% AuC \033[0m".format(auc_score * 100))
        print("-------------------------------------------------------\n\n")
        self.log('AUC', auc_score, sync_dist=True)

        return auc_score

    def test_step(self, batch:List[torch.Tensor], batch_idx:int) -> None:
        """
        test step of the model. It saves the output of the model and the input data as 
        List[torch.Tensor]: [predicted poses and the loss, tensor_data, transformation_idx, metadata, actual_frames]

        Args:
            batch (List[torch.Tensor]): list containing the following tensors:
                                        - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                        - transformation_idx
                                        - metadata
                                        - actual_frames
            batch_idx (int): index of the batch
        """

        x, target = batch
        pred = self(x)
        score = self.score_fn(pred, target)
        self._test_output_list.append(score.cpu().detach().numpy().squeeze())
        return


    def on_test_epoch_start(self) -> None:
        """
        Called when the test epoch begins.
        """
        
        super().on_test_epoch_start()
        self._test_output_list = []
        self.plot_images = []
        return


    def on_test_epoch_end(self) -> float:
        """
        test epoch end of the model.

        Returns:
            float: test auc score
        """
        scores = np.concatenate(self._test_output_list, axis=0)
        auc_score, scores_arr, _, _ =  self.post_processing(scores, True)
        print(scores_arr.shape)
        np.save(f'../final_score/{self.dataset}/skl_score_.npy', scores_arr)
        print("\n-------------------------------------------------------")
        print("\033[92m Done with {}% AuC for {} samples\033[0m".format(auc_score * 100, scores_arr.shape[0]))
        print("-------------------------------------------------------\n\n")

        return auc_score

    def post_processing(self, scores, test=False):
        gt_arr, scores_arr, clip_list = get_dataset_scores(scores, self.metadata, self.args)
        scores_arr = smooth_scores(scores_arr, 9)
        scores_np = MinMaxNorm(scores_arr)
        gt_np = np.concatenate(gt_arr)
        auc = roc_auc_score(gt_np, scores_np)
        return auc, scores_np, gt_arr, clip_list

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1, verbose=False)
        return {'optimizer':optimizer,'lr_scheduler':scheduler,'monitor':'AUC'}
        # return {'optimizer':optimizer,'monitor':'AUC'}