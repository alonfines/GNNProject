from torch_geometric.datasets import LRGBDataset
from torch_geometric.data import DataLoader


def load_data(batch_size=4):
    train_dataset = LRGBDataset(root='./data/Peptides-Func', name='Peptides-func', split='train')
    val_dataset = LRGBDataset(root='./data/Peptides-Func', name='Peptides-func', split='val')
    test_dataset = LRGBDataset(root='./data/Peptides-Func', name='Peptides-func', split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset












