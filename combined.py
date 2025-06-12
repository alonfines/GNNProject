import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool
from sklearn.metrics import average_precision_score
from pytorch_lightning.loggers import WandbLogger
import numpy as np

from dataset import load_peptides_func
from model import SpectralGCN       # your existing spectral model class
from prob_virtual_node import ProbabilisticVirtualNode  # your existing VN model class


class CombinedModel(LightningModule):
    def __init__(
        self,
        # pass through whatever hyper-params you need:
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        k: int,
        num_virtual_nodes: int,
        num_classes: int,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 2) instantiate your two LightningModules
        #    - set out_channels=num_classes on SpectralGCN
        self.spectral = SpectralGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            k=k,             # if your SpectralGCN takes k
            lr=lr
        )
        self.vn = ProbabilisticVirtualNode(
            in_dim=in_channels,        # same node-feature size
            hidden_dim=hidden_channels,
            num_virtual_nodes=num_virtual_nodes,
            k=k,
            num_classes=num_classes,
            lr=lr,
            num_layers=num_layers,
            dropout=dropout
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, data):
        # both expect a single `data` object
        spec_logits = self.spectral(data.x, data.edge_index, getattr(data, "batch", None))   # [B, num_classes]
        vn_logits   = self.vn(data)         # [B, num_classes]
        return spec_logits, vn_logits

    def training_step(self, batch, batch_idx):
        spec_logits, vn_logits = self(batch)
        y = batch.y.float()  # shape [B, num_classes]

        # 3) sum the two losses so both models get gradients
        loss_spec = self.criterion(spec_logits, y)
        loss_vn   = self.criterion(vn_logits,   y)
        loss      = loss_spec + loss_vn

        # Calculate average precision only if there are positive samples
        vn_probs = torch.sigmoid(vn_logits).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        
        # Check if there are any positive samples in the batch
        if np.any(y_true):
            avg_prec = average_precision_score(y_true, vn_probs, average='macro')
        else:
            avg_prec = 0.0  # or any other default value you prefer

        self.log("train/loss_spec", loss_spec, on_step=True, prog_bar=False, batch_size=batch.y.size(0))
        self.log("train/loss_vn",   loss_vn,   on_step=True, prog_bar=False, batch_size=batch.y.size(0))
        self.log("train/loss",      loss,      on_step=True, prog_bar=False, batch_size=batch.y.size(0))
        self.log("train/avg_precision", avg_prec, on_step=True, prog_bar=False, batch_size=batch.y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        _, vn_logits = self(batch)
        y = batch.y.float()
        val_loss = self.criterion(vn_logits, batch.y.float())

        vn_probs = torch.sigmoid(vn_logits).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        
        # Check if there are any positive samples in the batch
        if np.any(y_true):
            avg_prec = average_precision_score(y_true, vn_probs, average='macro')
        else:
            avg_prec = 0.0  # or any other default value you prefer

        self.log("val/loss", val_loss, prog_bar=False, batch_size=batch.y.size(0))
        self.log("val/avg_precision", avg_prec, prog_bar=False, batch_size=batch.y.size(0))
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    import wandb

    wandb_logger = WandbLogger(project="Combined")


    # load your peptide-func data exactly as you did before:
    train_loader, val_loader, in_dim, num_classes = load_peptides_func(batch_size=16)

    # instantiate with the same hyperparams you used before
    model = CombinedModel(
        in_channels=in_dim,
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
        k=4,
        num_virtual_nodes=4,
        num_classes=num_classes,
        lr=1e-4
    )

    from pytorch_lightning import Trainer
    trainer = Trainer(max_epochs=100, accelerator="auto", devices=1, precision=16, logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)