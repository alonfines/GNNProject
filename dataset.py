from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader

def load_peptides_func(batch_size=64):
    train_dataset = LRGBDataset(root='data', name='Peptides-func', split='train')
    val_dataset = LRGBDataset(root='data', name='Peptides-func', split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, train_dataset.num_node_features, train_dataset.num_classes
