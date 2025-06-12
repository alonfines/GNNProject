from models.prob_virtual_node import ProbabilisticVirtualNode
from data.dataset import load_peptides_func
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb

def main():
    train_loader, val_loader, num_features, num_classes = load_peptides_func()

    model = ProbabilisticVirtualNode(
        in_dim=num_features,
        hidden_dim=64,
        num_virtual_nodes=4,
        k=2,
        num_classes=num_classes,
        lr=0.001
    )

    wandb_logger = WandbLogger(project="Peptides-Prob-VN", name="GCN")
    #checkpoint = ModelCheckpoint(monitor="val/avg_precision", mode="max", save_top_k=1)
    # early_stop = EarlyStopping(monitor="val/avg_precision", mode="max", patience=20)

    trainer = Trainer(
        max_epochs=100,
        logger=wandb_logger,
        #callbacks=[checkpoint, early_stop],
        accelerator="auto",
        devices=1,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

if __name__ == '__main__':
    main()