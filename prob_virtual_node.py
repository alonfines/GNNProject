import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import dropout_edge
from pytorch_lightning import LightningModule
from sklearn.metrics import average_precision_score

class ProbabilisticVirtualNode(LightningModule):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_virtual_nodes: int = 4,
        k: int = 4,
        num_classes: int = 10,
        lr: float = 1e-3,
        num_layers: int = 3,
        dropout: float = 0.2,
        edge_drop_rate: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # projection, sampler MLP
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.logit_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_virtual_nodes),
        )

        # virtual-node embeddings (shared, but will replicate per graph)
        self.virtual_nodes = nn.Parameter(torch.randn(num_virtual_nodes, hidden_dim))

        # stack of GNN layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.dropout = dropout
        self.edge_drop_rate = edge_drop_rate

        # final head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _gumbel_topk(logits: torch.Tensor, k: int):
        """Straight‐through Gumbel‐TopK: returns one‐hot selections of size k per row."""
        # logits: [N, M]
        gumbels = -torch.empty_like(logits).exponential_().log()  # sample Gumbel noise
        noisy = (logits + gumbels)  # [N, M]
        topk = noisy.topk(k, dim=-1).indices  # [N, k]
        one_hot = torch.zeros_like(logits).scatter_(
            1, topk, 1.0
        )
        # straight-through: use one-hot forward, gradients to logits
        return one_hot + (logits - logits).detach()

    def forward(self, data):
        device = data.x.device
        x = data.x.float().to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        N = x.size(0)
        B = batch.max().item() + 1
        M = self.hparams.num_virtual_nodes
        k = self.hparams.k       

        # --- 1. predict connection logits & differentiate through sampling
        theta_logits = self.logit_mlp(x) 
        theta_probs = F.softmax(theta_logits, dim=-1)
        sel_matrix = self._gumbel_topk(theta_probs, k)  # [N, M]

        # --- 2. build virtual edges per node, per graph
        # for node i in graph g = batch[i], connect to virtual node indices:
        #   global index = N + g*M + v
        batch_idx = batch.unsqueeze(1)                       # [N, 1]
        virtual_offsets = (batch_idx * M).expand(-1, M)      # [N, M]
        sel_v = (sel_matrix * torch.arange(M, device=device)).long()  # [N, M]
        dst_virtual = N + virtual_offsets + sel_v            # [N, M]

        src_regular = torch.arange(N, device=device).unsqueeze(1).expand(-1, k)
        edge_src = torch.cat([src_regular.flatten(), dst_virtual.flatten()], dim=0)
        edge_dst = torch.cat([dst_virtual.flatten(), src_regular.flatten()], dim=0)
        virtual_edge_index = torch.stack([edge_src, edge_dst], dim=0)  # [2, 2*N*k]

        # --- 3. replicate virtual‐node features per graph
        # shape (B, M, hidden) -> (B*M, hidden)
        x_virtual = self.virtual_nodes.unsqueeze(0).expand(B, -1, -1)
        x_virtual = x_virtual.reshape(B * M, -1)

        # --- 4. combine features & edge_index
        x = F.relu(self.input_proj(x))
        x_ext = torch.cat([x, x_virtual], dim=0)

        # --- 5. optionally drop some original edges during training
        if self.training and self.edge_drop_rate > 0.0:
            edge_index, _ = dropout_edge(
                edge_index, p=self.edge_drop_rate, force_undirected=False
            )

        edge_index_ext = torch.cat([edge_index, virtual_edge_index], dim=1)

        # --- 6. multi‐layer message passing with residuals & dropout
        for conv in self.convs:
            out = conv(x_ext, edge_index_ext)
            out = F.dropout(out, p=self.dropout, training=self.training)
            x_ext = F.relu(x_ext + out)  # residual

        # --- 7. pool only real nodes
        x_real = x_ext[:N]
        graph_repr = global_add_pool(x_real, batch)

        return self.head(graph_repr)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch.y.float())
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        y_true = batch.y.cpu().numpy()
        y_scores = torch.sigmoid(logits).cpu().numpy()

        # compute AP exactly as in the published code
        ap_list = []
        for i in range(y_true.shape[1]):
            if y_true[:, i].sum() > 0 and (y_true[:, i] == 0).sum() > 0:
                ap = average_precision_score(y_true[:, i], y_scores[:, i])
                ap_list.append(ap)
        avg_ap = sum(ap_list) / len(ap_list)
        self.log("val/avg_precision", avg_ap, on_epoch=True, prog_bar=True, batch_size=batch.y.size(0))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max'),
            'monitor': 'val/avg_precision'
        }
        return [opt], [sched]
