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
from model import SpectralGCN
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
        train_mode: str = "combined",  # Options: "spectral", "vn", "combined"
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if train_mode not in ["spectral", "vn", "combined"]:
            raise ValueError("train_mode must be one of: 'spectral', 'vn', 'combined'")
        self.train_mode = train_mode

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
            in_dim=in_channels,
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
        if self.train_mode == "spectral":
            return self.spectral(data.x, data.edge_index, getattr(data, "batch", None)), None
        elif self.train_mode == "vn":
            return None, self.vn(data)
        else:
            spec_logits = self.spectral(data.x, data.edge_index, getattr(data, "batch", None))
            vn_logits = self.vn(data)
            return spec_logits, vn_logits

    def training_step(self, batch, batch_idx):
        spec_logits, vn_logits = self(batch)
        y = batch.y.float()  # shape [B, num_classes]

        # 3) sum the two losses so both models get gradients
        loss_spec = self.criterion(spec_logits, y)
        loss_vn   = self.criterion(vn_logits,   y)
        loss      = loss_spec + loss_vn

        vn_probs = torch.sigmoid(vn_logits).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        avg_prec = average_precision_score(y_true, vn_probs, average='macro')

        self.log("train/loss_spec", loss_spec, on_step=True, prog_bar=False, batch_size=batch.y.size(0))
        self.log("train/loss_vn",   loss_vn,   on_step=True, prog_bar=False, batch_size=batch.y.size(0))
        self.log("train/loss",      loss,      on_step=True, prog_bar=False, batch_size=batch.y.size(0))
        self.log("train/avg_precision", avg_prec, on_step=True, prog_bar=False, batch_size=batch.y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        spec_logits, vn_logits = self(batch)
        y = batch.y.float()
        y_true = y.detach().cpu().numpy()
        avg_prec = average_precision_score(y_true, vn_probs, average='macro')

        self.log("val/loss", val_loss, prog_bar=False, batch_size=batch.y.size(0))
        self.log("val/avg_precision", avg_prec, prog_bar=False, batch_size=batch.y.size(0))
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# === Training Script ===

if __name__ == "__main__":
    import wandb
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer

    wandb_logger = WandbLogger(project="Combined")

    train_loader, val_loader, in_dim, num_classes = load_peptides_func(batch_size=32)

    model = CombinedModel(
        in_channels=in_dim,
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
        k=4,
        num_virtual_nodes=4,
        num_classes=num_classes,
        train_mode="combined",  # Options: "spectral", "vn", "combined"
        lr=1e-4
    )

    monitor_metric = "val_spectral_avg_precision" if model.train_mode == "spectral" else "val_vn_avg_precision"

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath="GNNProject/checkpoints",
        filename=f"{model.train_mode}-best-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
        save_top_k=1,
        mode="max",
        save_last=True
    )

    trainer = Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        precision=16,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader)
    wandb.finish()
    print("Training complete.")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best {monitor_metric} value: {checkpoint_callback.best_model_score:.4f}")