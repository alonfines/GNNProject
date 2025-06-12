from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import wandb
import os
import torch

from model import GCN, SpectralGCN
from peptides_func_data import load_data


def train(model_type, lr=1e-3, batch_size=128, k=5, max_epochs=100, hidden_channels=64):
    # Initialize WandB
    wandb.init(project="gnn-project", name=f"{model_type}-training", reinit=True)
    wandb_logger = WandbLogger(project="gnn-project", log_model=True)

    # Load data
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(batch_size=batch_size)

    in_channels = train_dataset[0].x.size(1)
    out_channels = train_dataset[0].y.size(1)

    # Model and checkpoint callback
    if model_type == "GCN":
        model = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, lr=lr)
        monitor_metric = "val_ap"
        filename = "gcn_model_best_{epoch:02d}_{val_ap:.4f}"

    elif model_type == "SpectralGCN":
        model = SpectralGCN(in_channels=in_channels, hidden_channels=hidden_channels,
                            out_channels=out_channels, lr=lr, k=k)
        monitor_metric = "val_loss"
        filename = "spectral_gcn_model_best_{epoch:02d}_{val_loss:.4f}"

    elif model_type == "SimpleS2GNN":
        model = SimpleS2GNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=3,
            heads=4
        )
        monitor_metric = "val_loss"
        filename = "s2gnn_model_best_{epoch:02d}_{val_loss:.4f}"

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Common trainer components
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode="min" if "loss" in monitor_metric else "max",
        save_top_k=1,
        dirpath=os.path.dirname(os.path.abspath(__file__)),
        filename=filename
    )

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=10,
        mode="min" if "loss" in monitor_metric else "max"
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    print(f"\n✅ Best model saved at: {checkpoint_callback.best_model_path}")
    wandb.finish()


if __name__ == "__main__":
    train(
        model_type="SimpleS2GNN",  # Change to "GCN" or "SpectralGCN"
        lr=1e-4,
        batch_size=32,
        max_epochs=100,
        hidden_channels=64,
        k=6
    )
