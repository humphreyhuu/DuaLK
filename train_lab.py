import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
import pickle
import numpy as np

from metrics import top_k_prec_recall


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):
    ks = [20, 40]
    best_test_loss = float('inf')
    best_model_path = './saved/train_lab/lab_layer_checkpoint.pth'

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0

        y_true_test = []
        y_pred_test = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                y_true_test.append(labels.cpu().numpy())
                y_pred_test.append(outputs.cpu().numpy())

        y_true_test = np.vstack(y_true_test)
        y_pred_test = np.vstack(y_pred_test)
        y_pred_sorted_test = np.argsort(y_pred_test, axis=1)[:, ::-1]

        _, recall_at_k_test = top_k_prec_recall(y_true_test, y_pred_sorted_test, ks)
        test_recall20 = recall_at_k_test[0]
        test_recall40 = recall_at_k_test[1]

        test_loss /= len(test_loader)

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, '
              f'Test Recall@20: {test_recall20:.4f}, '
              f'Test Recall@40: {test_recall40:.4f}, '
              f'Learning Rate: {current_lr:.6f}')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'linear1_weight': model[0].weight.data,
                'linear1_bias': model[0].bias.data,
                'linear2_weight': model[3].weight.data,
                'linear2_bias': model[3].bias.data,
            }, best_model_path)
            print(f'Checkpoint saved at epoch {epoch + 1} with test loss: {test_loss:.4f}')


if __name__ == '__main__':
    data_path = 'data'
    dataset = 'mimic3'  # 'mimic4'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Loading the prerequisite data...')
    dataset_path = os.path.join(data_path, dataset)
    standard_path = os.path.join(dataset_path, 'standard')
    labs_dataset = pickle.load(open(os.path.join(standard_path, 'labs_dataset.pkl'), 'rb'))
    train_labs_x, train_labs_y = labs_dataset['train_labs_data']
    test_labs_x, test_labs_y = labs_dataset['test_labs_data']

    train_labs_x, train_labs_y = torch.from_numpy(train_labs_x), torch.from_numpy(train_labs_y)
    test_labs_x, test_labs_y = torch.from_numpy(test_labs_x), torch.from_numpy(test_labs_y)
    print(train_labs_x.shape, train_labs_y.shape)
    print(test_labs_x.shape, test_labs_y.shape)

    item_num = train_labs_x.shape[1]
    code_num = train_labs_y.shape[1]

    model = nn.Sequential(
        nn.Linear(item_num, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, code_num),
    ).to(device)

    train_dataset = TensorDataset(train_labs_x.float(), train_labs_y.float())
    test_dataset = TensorDataset(test_labs_x.float(), test_labs_y.float())
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    num_epochs = 100
    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device)

    end_model_path = './saved/train_lab/lab_layer_checkpoint_end.pth'
    torch.save({
        'linear1_weight': model[0].weight.data,
        'linear1_bias': model[0].bias.data,
        'linear2_weight': model[3].weight.data,
        'linear2_bias': model[3].bias.data,
    }, end_model_path)
    print('Saving the final checkpoint...')
