import numpy as np
import random

import warnings
from sklearn.exceptions import UndefinedMetricWarning

import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader

from models.model import DuaLK
from utils import PatientLabDataset, load_data
from utils import pretrain_model_jointly, pretrain_individual_decoder


model_config = {
    'init_dim': 2000,
    'GNN': {
        'type': 'gat',  # ['gcn', 'gat']
        'dims': (256, 256),
        'dropout': 0.,
    },
    'Attention': {
        'dropout': 0.2,
    },
    'Decoder': {
        'dims': (256, 128),
        'dropout': 0.4,
    },
    'Classifier': {
        'dims': [256],
        'dropout': 0.4,
    }
}


if __name__ == '__main__':
    data_path, dataset = 'data', 'mimic3'
    pretrain = True  # Not allowed to change
    train_type = 'pretrain'  # ['direct', 'pretrain', 'finetune']
    use_lab = False  # Not allowed to change
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    code_fuse, visit_fuse = 'simple', 'simple'
    gnn_type = model_config['GNN']['type']  # ['gcn', 'gat']

    '''Special for this script'''
    step = 'together'  # ['joint', 'individual', 'together']
    random_seed = 66
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    train_codes_x, pretrain_codes_y, edge_index, x, edge_weight = load_data(pretrain, data_path, dataset,
                                                                            model_config['init_dim'])
    _, pretrain_codes_y_hema, pretrain_codes_y_chem, pretrain_codes_y_blood = pretrain_codes_y

    x = x.float()
    data = Data(x=x, edge_index=edge_index)  # , edge_weight=edge_weight
    print(data.x.shape)

    print('The shape of the training data is:', train_codes_x.shape)
    train_dataset = PatientLabDataset(train_codes_x, pretrain_codes_y_hema,
                                      pretrain_codes_y_chem, pretrain_codes_y_blood)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_indices = np.random.choice(len(train_dataset), 10000, replace=False)
    test_subset = torch.utils.data.Subset(train_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)

    learning_rate = 0.001
    joint_epochs = 3  # Set 3 for quick start, 10 for full training
    individual_epochs = 3  # Set 3 for quick start, 10 for full training
    num_classes = [pretrain_codes_y_hema.shape[1], pretrain_codes_y_chem.shape[1], pretrain_codes_y_blood.shape[1]]

    print('The current device is:', device)
    data = data.to(str(device))
    model = DuaLK(model_config=model_config, emb_init=data.x, num_classes=num_classes,
                    use_lab=use_lab, code_fuse=code_fuse, visit_fuse=visit_fuse, train_type=train_type,
                    lab_weight=None, lab_bias=None, gnn_type=gnn_type).to(device)
    print(model)

    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if step == "joint" or step == 'together':
        print('Joint Pretraining...')
        pretrain_model_jointly(model, data, train_loader, criterion, optimizer, joint_epochs, device, test_loader)
        torch.save(model.state_dict(), f'./saved/joint_pretrained_model_{joint_epochs}.pth')
    if step == "individual" or step == 'together':
        model.load_state_dict(torch.load(f'./saved/joint_pretrained_model_{joint_epochs}.pth'))
        print('Pretraining Individual Decoders...')
        decoders = ['hema', 'chem', 'blood']
        for decoder_type in decoders:
            pretrain_individual_decoder(model, data, train_loader, criterion, optimizer, individual_epochs, device,
                                        decoder_type, test_loader)
