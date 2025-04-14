import pickle

import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning

import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader

from models.model import DualMAR
from utils import PatientDataset, load_data, get_rare_data, exclude_chronic_codes, drop_empty_admissions
from metrics import top_k_prec_recall, f1
from models.loss import WeightedBCEWithLogitsLoss


def adjust_learning_rate(optimizer, epoch):
    if epoch > 250:
        lr = 0.000001
    elif epoch > 200:
        lr = 0.00001
    elif epoch > 150:
        lr = 0.00005
    elif epoch > 100:
        lr = 0.0001
    elif epoch > 50:
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
        'dropout': 0.4,
    }
}


if __name__ == '__main__':
    data_path, dataset = 'data', 'mimic3'
    pretrain = False  # Not allowed to change
    train_type = 'finetune'  # ['direct', 'pretrain', 'finetune']
    use_lab = True  # ['True', 'False']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    code_fuse, visit_fuse = 'simple', 'simple'
    loss = 'pos_weight'  # ['label_weight', 'pos_weight'], main: 'pos_weight'
    gnn_type = model_config['GNN']['type']  # ['gcn', 'gat'], main: 'gat'
    code_range = 'all'  # ['rare', 'all'], main: 'all'
    exclude_chronic = False  # [True, False], main: False

    if code_range == 'rare':
        min_freq, max_freq = 5, 20  # 0, 10
        sample_num = 200  # 200
    else:
        min_freq, max_freq, sample_num = None, None, None

    direct_train = True  # Deprecated
    if loss == 'label_weight':
        pos_weight, neg_weight = None, None
        print('Using label_weight loss...')
    elif loss == 'pos_weight':
        pos_weight, neg_weight = 1, 1
        print('Pos_weight:', pos_weight, 'Neg_weight:', neg_weight)
    else:
        raise ValueError('Invalid loss type')

    (train_codes_x, train_codes_y, test_codes_x, test_codes_y, train_labs_x, test_labs_x,
     edge_index, x, edge_weight, train_codes_y_r, test_codes_y_r) = load_data(pretrain, data_path, dataset,
                                                                              model_config['init_dim'])
    x = x.float()
    data = Data(x=x, edge_index=edge_index)  # , edge_weight=edge_weight
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    if code_range == 'all':
        train_codes_y, test_codes_y = train_codes_y, test_codes_y
    elif code_range == 'rare':
        print('Only focus on rare codes...')
        train_codes_x, train_codes_y, test_codes_x, test_codes_y = get_rare_data(train_codes_x, train_codes_y,
                                                                                 test_codes_x, test_codes_y,
                                                                                 min_freq, max_freq, sample_num)
        print('Less Frequent Data Shape: ', train_codes_y.shape, test_codes_y.shape)

    if code_range == 'all' and exclude_chronic:
        with open('./data/chronic/chronic_pos.pkl', 'rb') as f:
            chronic_pos = pickle.load(f)
        print('Excluding chronic codes...')
        train_codes_y, test_codes_y = exclude_chronic_codes(train_codes_y, test_codes_y, chronic_pos)
        train_codes_x, train_codes_y, test_codes_x, test_codes_y = drop_empty_admissions(train_codes_x, train_codes_y,
                                                                                         test_codes_x, test_codes_y)
        print(train_codes_y.shape, test_codes_y.shape)

    train_dataset = PatientDataset(train_codes_x, train_labs_x.float(), train_codes_y)
    test_dataset = PatientDataset(test_codes_x, test_labs_x.float(), test_codes_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    learning_rate = 0.001
    epochs = 500
    if dataset == 'mimic3':
        num_classes = [159, 115, 16, train_codes_y.shape[1]]
    elif dataset == 'mimic4':
        num_classes = [239, 191, 15, train_codes_y.shape[1]]
    else:
        num_classes = []
        raise ValueError('Invalid dataset name')

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
    print(model)

    if train_type in ['pretrain', 'finetune']:
        print('Loading pretrained parameters...')
        model.load_partial_state_dict(torch.load('./saved/joint_pretrained_model_10.pth', map_location=device))
        model.load_decoder_weights('./saved/decoder_hema_weights.pth',
                                   './saved/decoder_chem_weights.pth',
                                   './saved/decoder_blood_weights.pth', device)
        model = model.to(device)

    # torch.nn.BCEWithLogitsLoss().to(device)
    if loss == 'label_weight':
        total_samples = train_codes_y.size(0)
        num_classes = train_codes_y.size(1)
        pos_counts = train_codes_y.sum(dim=0)
        weights = total_samples / (pos_counts * num_classes + 1e-10)
        min_weight = weights.min()
        max_weight = weights.max()
        scaled_weights = 1 + (weights - min_weight) * (5 - 1) / (max_weight - min_weight + 1e-10)
        criterion = torch.nn.BCEWithLogitsLoss(weight=scaled_weights).to(device)
    elif loss == 'pos_weight':
        criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weight, neg_weight=neg_weight).to(device)
    else:
        raise ValueError('Invalid loss type')

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
            labels = labels.to(device)

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
                labels = labels.to(device)

                output = model(data, patient_data)
                loss = criterion(output, labels.float())
                test_loss += loss.item()

                y_true.append(labels.cpu().numpy())
                y_pred.append(output.cpu().numpy())

        test_loss /= len(test_loader)
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        y_pred_sorted = np.argsort(y_pred, axis=1)[:, ::-1]

        f1_weighted = f1(y_true, y_pred_sorted, metrics='weighted')
        ks = [10, 20, 30, 40] if code_range == 'all' else [5, 8, 15]
        _, recall_at_k = top_k_prec_recall(y_true, y_pred_sorted, ks)

        recall_str = ", ".join([f"Recall@{k}: {recall:.4f}" for k, recall in zip(ks, recall_at_k)])
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}, '
              f'F1-weighted: {f1_weighted:.4f}, {recall_str}, Learning Rate: {current_lr:.6f}')
