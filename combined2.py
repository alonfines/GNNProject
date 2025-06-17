import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from sklearn.metrics import average_precision_score
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from dataset import load_peptides_func
from model2 import SpectralGCN
from prob_virtual_node import ProbabilisticVirtualNode


class CombinedModel(LightningModule):
    def __init__(
        self,
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

        self.spectral = SpectralGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            k=k,
            lr=lr
        )
        self.vn = ProbabilisticVirtualNode(
            in_dim=hidden_channels,
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
        spec_embeddings = self.spectral(
            data.x, data.edge_index, getattr(data, "batch", None), return_embeddings=True
        )  # [num_nodes, hidden_channels]

        data_for_vn = Data(
            x=spec_embeddings,
            edge_index=data.edge_index,
            batch=getattr(data, "batch", None),
            y=data.y
        )

        vn_logits = self.vn(data_for_vn)  # [batch_size, num_classes]

        spec_logits = self.spectral(
            data.x, data.edge_index, getattr(data, "batch", None), return_embeddings=False
        )  # [batch_size, num_classes]

        return spec_logits, vn_logits

    def training_step(self, batch, batch_idx):
        spec_logits, vn_logits = self(batch)
        y = batch.y.float()

        loss_spec = self.criterion(spec_logits, y)
        loss_vn = self.criterion(vn_logits, y)
        loss = loss_spec + loss_vn

        vn_probs = torch.sigmoid(vn_logits).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        avg_prec = average_precision_score(y_true, vn_probs, average="macro")

        self.log("train/loss_spec", loss_spec, on_step=True, prog_bar=False, batch_size=y.size(0))
        self.log("train/loss_vn", loss_vn, on_step=True, prog_bar=False, batch_size=y.size(0))
        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=y.size(0))
        self.log("train/avg_precision", avg_prec, on_step=True, prog_bar=True, batch_size=y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        _, vn_logits = self(batch)
        y = batch.y.float()
        val_loss = self.criterion(vn_logits, y)

        vn_probs = torch.sigmoid(vn_logits).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        avg_prec = average_precision_score(y_true, vn_probs, average="macro")

        self.log("val/loss", val_loss, prog_bar=True, batch_size=y.size(0))
        self.log("val/avg_precision", avg_prec, prog_bar=True, batch_size=y.size(0))
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="Combined")

    train_loader, val_loader, in_dim, num_classes = load_peptides_func(batch_size=16)

    model = CombinedModel(
        in_channels=in_dim,
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
        k=4,
        num_virtual_nodes=4,
        num_classes=num_classes,
        lr=1e-4,
    )

    trainer = Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=wandb_logger,
    )
    trainer.fit(model, train_loader, val_loader)
