import numpy as np
import os
import pickle
import time

import torch
from torch.utils.data import Dataset

from metrics import f1


class PatientDataset(Dataset):
    def __init__(self, codes_x, labs_x, labels):
        self.codes_x = codes_x
        self.labs_x = labs_x
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'codes_x': self.codes_x[idx], 'labs_x': self.labs_x[idx]}
        return sample, self.labels[idx]


class PatientLabDataset(Dataset):
    def __init__(self, codes_x, labels_hema, labels_chem, labels_blood):
        self.codes_x = codes_x
        self.labels_hema = labels_hema
        self.labels_chem = labels_chem
        self.labels_blood = labels_blood

    def __len__(self):
        return len(self.labels_hema)

    def __getitem__(self, idx):
        sample = {'codes_x': self.codes_x[idx]}
        return (sample, self.labels_hema[idx],
                self.labels_chem[idx], self.labels_blood[idx])


def get_rare_data(train_codes_x, train_codes_y,
                  test_codes_x, test_codes_y,
                  min_threshold=5, max_threshold=10,
                  code_num=100, random_seed=99):
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # Step 1: Calculate frequency of each label in train_codes_y
    label_frequencies = torch.sum(train_codes_y, dim=0)  # Sum over samples, get frequency for each label
    # Step 2: Identify indices of labels within the specified thresholds
    valid_indices = (label_frequencies >= min_threshold) & (label_frequencies <= max_threshold)
    valid_label_indices = torch.where(valid_indices)[0]
    # Step 3: Randomly select a subset of labels if necessary
    if len(valid_label_indices) > code_num:
        sampled_indices = np.random.choice(valid_label_indices.cpu().numpy(), code_num, replace=False)
        sampled_indices = torch.tensor(sampled_indices, dtype=torch.long)
    else:
        sampled_indices = valid_label_indices
    # Step 4: Filter train_codes_y and test_codes_y to include only the sampled labels
    train_codes_y_sampled = train_codes_y[:, sampled_indices]
    test_codes_y_sampled = test_codes_y[:, sampled_indices]
    # Step 5: Identify non-zero rows (samples with at least one positive label)
    train_nonzero_indices = torch.where(torch.sum(train_codes_y_sampled, dim=1) > 0)[0]
    test_nonzero_indices = torch.where(torch.sum(test_codes_y_sampled, dim=1) > 0)[0]
    # Step 6: Filter train_codes_x, train_codes_y_sampled, test_codes_x, and test_codes_y_sampled
    train_codes_x_filtered = train_codes_x[train_nonzero_indices]
    train_codes_y_filtered = train_codes_y_sampled[train_nonzero_indices]
    test_codes_x_filtered = test_codes_x[test_nonzero_indices]
    test_codes_y_filtered = test_codes_y_sampled[test_nonzero_indices]
    # Return the processed datasets
    return train_codes_x_filtered, train_codes_y_filtered, test_codes_x_filtered, test_codes_y_filtered


# def adjust_learning_rate(optimizer, epoch):
#     base_lr = optimizer.param_groups[0]['lr']
#     if epoch >= 30:
#         lr = base_lr * 0.125
#     elif epoch >= 20:
#         lr = base_lr * 0.25
#     elif epoch >= 10:
#         lr = base_lr * 0.5
#     else:
#         lr = base_lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
def adjust_learning_rate(optimizer, epoch):
    base_lr = optimizer.param_groups[0]['lr']
    if epoch in [10, 20, 30, 40, 50]:
        lr = base_lr * 0.5
    else:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_data(pretrain: bool, data_path: str, dataset: str, init_dim: int = 2000, task: str = 'code'):
    dataset_path = os.path.join(data_path, dataset)
    standard_path = os.path.join(dataset_path, 'standard')
    graph_path = os.path.join(dataset_path, 'graph')

    disease2disease = pickle.load(open(os.path.join(graph_path, 'disease2disease.pkl'), 'rb'))
    edge_index, edge_weight, x = (disease2disease['edge_index'], disease2disease['edge_weight'],
                                  disease2disease[f'x_hake_{init_dim}'])

    if pretrain:
        pretrain_codes_data = pickle.load(open(os.path.join(standard_path, 'pretrain_codes_dataset.pkl'), 'rb'))
        pretrain_codes_x, pretrain_codes_y, _, _ = pretrain_codes_data
        return pretrain_codes_x, pretrain_codes_y, edge_index, x, edge_weight
    else:
        if task == 'code':
            codes_dataset = pickle.load(open(os.path.join(standard_path, 'codes_dataset.pkl'), 'rb'))
            train_codes_data = codes_dataset['train_codes_data']
            test_codes_data = codes_dataset['test_codes_data']
            train_codes_x, train_codes_y, train_labs_x, train_codes_y_r = train_codes_data
            test_codes_x, test_codes_y, test_labs_x, test_codes_y_r = test_codes_data
            train_codes_x, train_codes_y = torch.from_numpy(train_codes_x), torch.from_numpy(train_codes_y)
            test_codes_x, test_codes_y = torch.from_numpy(test_codes_x), torch.from_numpy(test_codes_y)
            train_labs_x, test_labs_x = torch.from_numpy(train_labs_x), torch.from_numpy(test_labs_x)
            train_codes_y_r, test_codes_y_r = torch.from_numpy(train_codes_y_r), torch.from_numpy(test_codes_y_r)
            return (train_codes_x, train_codes_y, test_codes_x, test_codes_y, train_labs_x, test_labs_x,
                    edge_index, x, edge_weight, train_codes_y_r, test_codes_y_r)
        elif task == 'hf':
            codes_dataset = pickle.load(open(os.path.join(standard_path, 'codes_dataset.pkl'), 'rb'))
            train_codes_data = codes_dataset['train_codes_data']
            test_codes_data = codes_dataset['test_codes_data']
            train_codes_x, _, train_labs_x, _ = train_codes_data
            test_codes_x, _, test_labs_x, _ = test_codes_data
            heart_failure = pickle.load(open(os.path.join(standard_path, 'heart_failure.pkl'), 'rb'))
            train_codes_y = heart_failure['train_hf_y']
            test_codes_y = heart_failure['test_hf_y']
            train_codes_x, train_codes_y = torch.from_numpy(train_codes_x), torch.from_numpy(train_codes_y)
            test_codes_x, test_codes_y = torch.from_numpy(test_codes_x), torch.from_numpy(test_codes_y)
            train_labs_x, test_labs_x = torch.from_numpy(train_labs_x), torch.from_numpy(test_labs_x)
            return (train_codes_x, train_codes_y, test_codes_x, test_codes_y, train_labs_x, test_labs_x,
                    edge_index, x, edge_weight)
        else:
            raise ValueError('Invalid task')


def exclude_chronic_codes(train_y, test_y, chronic_pos):
    # Convert chronic_pos from 1-based to 0-based by subtracting 1
    indices_to_remove = [pos - 1 for pos in chronic_pos]
    # Convert the list to a PyTorch tensor
    indices_to_remove = torch.tensor(indices_to_remove, dtype=torch.long)
    # Get all indices for the columns of y
    all_indices = torch.arange(train_y.size(1), dtype=torch.long)
    # Determine the indices that are not in indices_to_remove
    indices_to_keep = torch.tensor([idx for idx in all_indices if idx not in indices_to_remove])
    # Use the indices_to_keep to select the columns from both train and test tensors
    train_y_filtered = train_y[:, indices_to_keep]
    test_y_filtered = test_y[:, indices_to_keep]
    return train_y_filtered, test_y_filtered


def drop_empty_admissions(train_x, train_y, test_x, test_y):
    def filter_admissions(x, y):
        non_empty_samples = torch.any(y == 1, dim=1)
        filtered_x = x[non_empty_samples]
        filtered_y = y[non_empty_samples]
        return filtered_x, filtered_y
    train_codes_x, train_codes_y = filter_admissions(train_x, train_y)
    test_codes_x, test_codes_y = filter_admissions(test_x, test_y)

    return train_codes_x, train_codes_y, test_codes_x, test_codes_y


def pretrain_model_jointly(model, data, data_loader, criterion, optimizer, epochs, device, test_loader):

    model.train()
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_loss = 0

        train_start_time = time.time()

        for batch in data_loader:
            patient_data, lab1_batch, lab2_batch, lab3_batch = batch
            # patient_data = patient_data.to(device)
            patient_data = {k: v.to(device) for k, v in patient_data.items()}
            lab1_batch = lab1_batch.to(device)
            lab2_batch = lab2_batch.to(device)
            lab3_batch = lab3_batch.to(device)

            optimizer.zero_grad()
            output1, output2, output3 = model(data, patient_data)
            loss1 = criterion(output1, lab1_batch.float())
            loss2 = criterion(output2, lab2_batch.float())
            loss3 = criterion(output3, lab3_batch.float())
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(data_loader)

        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        eval_start_time = time.time()

        model.eval()
        test_loss = 0
        y_true1, y_true2, y_true3 = [], [], []
        y_pred1, y_pred2, y_pred3 = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                patient_data, lab1_batch, lab2_batch, lab3_batch = batch
                # patient_data = patient_data.to(device)
                patient_data = {k: v.to(device) for k, v in patient_data.items()}
                lab1_batch = lab1_batch.to(device)
                lab2_batch = lab2_batch.to(device)
                lab3_batch = lab3_batch.to(device)

                output1, output2, output3 = model(data, patient_data)
                loss1 = criterion(output1, lab1_batch.float())
                loss2 = criterion(output2, lab2_batch.float())
                loss3 = criterion(output3, lab3_batch.float())
                test_loss += (loss1.item() + loss2.item() + loss3.item())

                y_true1.append(lab1_batch.cpu().numpy())
                y_true2.append(lab2_batch.cpu().numpy())
                y_true3.append(lab3_batch.cpu().numpy())
                y_pred1.append(output1.cpu().numpy())
                y_pred2.append(output2.cpu().numpy())
                y_pred3.append(output3.cpu().numpy())

        test_loss /= len(test_loader)
        y_true1, y_true2, y_true3 = np.vstack(y_true1), np.vstack(y_true2), np.vstack(y_true3)
        y_pred1, y_pred2, y_pred3 = np.vstack(y_pred1), np.vstack(y_pred2), np.vstack(y_pred3)
        y_pred_sort1 = np.argsort(y_pred1, axis=1)[:, ::-1]
        y_pred_sort2 = np.argsort(y_pred2, axis=1)[:, ::-1]
        y_pred_sort3 = np.argsort(y_pred3, axis=1)[:, ::-1]

        f1_weighted1 = f1(y_true1, y_pred_sort1, metrics='weighted')
        f1_weighted2 = f1(y_true2, y_pred_sort2, metrics='weighted')
        f1_weighted3 = f1(y_true3, y_pred_sort3, metrics='weighted')

        eval_end_time = time.time()
        eval_duration = eval_end_time - eval_start_time

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'F1-weighted1: {f1_weighted1:.4f}, F1-weighted2: {f1_weighted2:.4f}, F1-weighted3: {f1_weighted3:.4f}, '
              f'Learning Rate: {current_lr:.5f}, Train Time: {train_duration:.2f}s, Eval Time: {eval_duration:.2f}s')


def pretrain_individual_decoder(model, data, data_loader, criterion, optimizer, epochs,
                                device, decoder_type, test_loader):
    model.train()

    for param in model.gnn_layer.parameters():
        param.requires_grad = False
    for param in model.visit_attention.parameters():
        param.requires_grad = False
    for param in model.patient_attention.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        # adjust_learning_rate(optimizer, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_loss = 0

        train_start_time = time.time()

        for batch in data_loader:
            patient_data, lab1_batch, lab2_batch, lab3_batch = batch
            # patient_data = patient_data.to(device)
            patient_data = {k: v.to(device) for k, v in patient_data.items()}
            lab1_batch = lab1_batch.to(device)
            lab2_batch = lab2_batch.to(device)
            lab3_batch = lab3_batch.to(device)

            optimizer.zero_grad()
            output1, output2, output3 = model(data, patient_data)

            if decoder_type == 'hema':
                loss = criterion(output1, lab1_batch.float())
            elif decoder_type == 'chem':
                loss = criterion(output2, lab2_batch.float())
            elif decoder_type == 'blood':
                loss = criterion(output3, lab3_batch.float())
            else:
                raise ValueError('Invalid decoder type')

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(data_loader)

        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        eval_start_time = time.time()

        model.eval()
        test_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                patient_data, lab1_batch, lab2_batch, lab3_batch = batch
                # patient_data = patient_data.to(device)
                patient_data = {k: v.to(device) for k, v in patient_data.items()}
                lab1_batch = lab1_batch.to(device)
                lab2_batch = lab2_batch.to(device)
                lab3_batch = lab3_batch.to(device)

                output1, output2, output3 = model(data, patient_data)

                if decoder_type == 'hema':
                    loss = criterion(output1, lab1_batch.float())
                    y_true.append(lab1_batch.cpu().numpy())
                    y_pred.append(output1.cpu().numpy())
                elif decoder_type == 'chem':
                    loss = criterion(output2, lab2_batch.float())
                    y_true.append(lab2_batch.cpu().numpy())
                    y_pred.append(output2.cpu().numpy())
                elif decoder_type == 'blood':
                    loss = criterion(output3, lab3_batch.float())
                    y_true.append(lab3_batch.cpu().numpy())
                    y_pred.append(output3.cpu().numpy())

                test_loss += loss.item()

        test_loss /= len(test_loader)
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        y_pred_sort = np.argsort(y_pred, axis=1)[:, ::-1]
        f1_weighted = f1(y_true, y_pred_sort, metrics='weighted')

        eval_end_time = time.time()
        eval_duration = eval_end_time - eval_start_time

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss ({decoder_type}): {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'F1-weighted: {f1_weighted:.4f}, Learning Rate: {current_lr:.5f}, '
              f'Train Time: {train_duration:.2f}s, Eval Time: {eval_duration:.2f}s')

        if decoder_type == 'hema':
            torch.save(model.decoder1.state_dict(), './saved/decoder_hema_weights.pth')
        elif decoder_type == 'chem':
            torch.save(model.decoder2.state_dict(), './saved/decoder_chem_weights.pth')
        elif decoder_type == 'blood':
            torch.save(model.decoder3.state_dict(), './saved/decoder_blood_weights.pth')
