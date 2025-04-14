import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning

import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader

from models.model import DualMAR
from utils import PatientDataset, load_data
from sklearn.metrics import f1_score, roc_auc_score
# from models.loss import WeightedBCEWithLogitsLoss
import torch.nn.functional as F


def adjust_learning_rate(optimizer, epoch):
    if epoch > 25:
        lr = 0.000001
    elif epoch > 20:
        lr = 0.00001
    elif epoch > 15:
        lr = 0.00005
    elif epoch > 10:
        lr = 0.0001
    elif epoch > 5:
        lr = 0.0005
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
        'dropout': 0.6,
    }
}


if __name__ == '__main__':
    data_path, dataset = 'data', 'mimic3'
    pretrain = False  # Not allowed to change
    train_type = 'finetune'  # ['direct', 'pretrain', 'finetune']
    use_lab = False  # ['True', 'False']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    code_fuse, visit_fuse = 'simple', 'simple'
    gnn_type = model_config['GNN']['type']  # ['gcn', 'gat']

    direct_train = True  # Deprecated

    (train_codes_x, train_codes_y, test_codes_x, test_codes_y, train_labs_x, test_labs_x,
     edge_index, x, edge_weight) = load_data(pretrain, data_path, dataset,
                                             model_config['init_dim'], task='hf')
    x = x.float()
    data = Data(x=x, edge_index=edge_index)  # , edge_weight=edge_weight
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    train_dataset = PatientDataset(train_codes_x, train_labs_x.float(), train_codes_y)
    test_dataset = PatientDataset(test_codes_x, test_labs_x.float(), test_codes_y)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    learning_rate = 0.001
    epochs = 50
    num_classes = [159, 115, 16, 1]  # train_codes_y.shape[1]

    if use_lab:
        print('Loading weights and bias from train_lab.py')
        model_path = './saved/train_lab/lab_layer_checkpoint_end.pth'
        checkpoint = torch.load(model_path)
        lab_weight, lab_bias = checkpoint['linear1_weight'], checkpoint['linear1_bias']
    else:
        lab_weight, lab_bias = None, None

    print('The current device is:', device)
    data = data.to(str(device))
    model = DualMAR(model_config=model_config, emb_init=data.x, num_classes=num_classes,
                    use_lab=use_lab, code_fuse=code_fuse, visit_fuse=visit_fuse, train_type=train_type,
                    lab_weight=lab_weight, lab_bias=lab_bias, gnn_type=gnn_type).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    print(model)

    if train_type in ['pretrain', 'finetune']:
        print('Loading pretrained parameters...')
        model.load_partial_state_dict(torch.load('./saved/joint_pretrained_model_10.pth'))
        model.load_decoder_weights('./saved/decoder_hema_weights.pth',
                                   './saved/decoder_chem_weights.pth',
                                   './saved/decoder_blood_weights.pth')
        model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    print('Training...')
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_loss = 0

        model.train()
        for batch in train_loader:
            patient_data, labels = batch
            # patient_data = patient_data.to(device)
            patient_data = {k: v.to(device) for k, v in patient_data.items()}
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            output = model(data, patient_data)
            loss = criterion(output, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)

        # Evaluation on test set
        model.eval()
        test_loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in test_loader:
                patient_data, labels = batch
                # patient_data = patient_data.to(device)
                patient_data = {k: v.to(device) for k, v in patient_data.items()}
                labels = labels.to(device).unsqueeze(1)

                output = model(data, patient_data)
                loss = criterion(output, labels.float())
                test_loss += loss.item()

                y_true.append(labels.cpu().numpy())
                y_pred.append(output.cpu().numpy())

        test_loss /= len(test_loader)
        y_true = np.concatenate(y_true).squeeze()
        y_pred = np.concatenate(y_pred).squeeze()

        y_pred_sigmoid = F.sigmoid(torch.tensor(y_pred)).numpy()
        y_pred_bin = (y_pred_sigmoid > 0.5).astype(int)
        auc = roc_auc_score(y_true, y_pred_sigmoid)
        f1 = f1_score(y_true, y_pred_bin)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}, '
              f'AUC: {auc:.4f}, F1 Score: {f1:.4f}, Learning Rate: {current_lr:.6f}')
