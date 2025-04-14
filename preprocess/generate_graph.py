import os
import pickle
import numpy as np

from collections import defaultdict
from typing import Tuple

import torch


def generate_disease_complication_edge_index(pids: np.ndarray, patient_admissions: dict,
                                             admissions_codes_encoded: dict, code_num: int,
                                             self_edge: bool = False,
                                             edge_score: float = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    co_occurrence_count = defaultdict(int)
    total_edges_per_code = defaultdict(int)
    print('Generating disease-disease complication edge index...')
    print('Vocabulary Size of codes:', code_num)

    for pid in pids:
        admissions = patient_admissions.get(pid, [])
        for admission in admissions:
            admission_id = admission['admission_id']
            codes = admissions_codes_encoded.get(admission_id, [])
            unique_codes = set(codes)
            for code in unique_codes:
                total_edges_per_code[code] += len(codes) - 1  # Update total edges count, subtracting self-occurrences
            for i in range(len(codes)):
                for j in range(i + 1, len(codes)):
                    code_a = codes[i]
                    code_b = codes[j]
                    co_occurrence_count[(code_a, code_b)] += 1
                    co_occurrence_count[(code_b, code_a)] += 1
                    if self_edge:  # For self edge
                        co_occurrence_count[(code_a, code_a)] += 1
                        co_occurrence_count[(code_b, code_b)] += 1

    edge_index = [[], []]
    edge_weight = []

    for (code_a, code_b), count in co_occurrence_count.items():
        edge_index[0].append(code_a)
        edge_index[1].append(code_b)
        if code_a == code_b:
            edge_weight.append(edge_score)  # Self-loop weight
        else:
            weight = count / total_edges_per_code[code_a]
            edge_weight.append(weight)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    return edge_index, edge_weight


def generate_disease_complication_x(icd2emb, icd_map, emb_dim=1536) -> torch.Tensor:
    # print('Generating disease-disease embedding init...')
    cannot_find = 0
    x = np.zeros(shape=(len(icd_map) + 1, emb_dim), dtype=float)
    for icd, pos in icd_map.items():
        emb = icd2emb.get(icd, None)
        if emb is None:
            # print('Warning 1: ICD code not found in the embedding:', icd)
            emb = icd2emb.get(icd[:-1], None)
            if emb is None:
                # print('Warning 2: ICD code not found in the embedding:', icd[:-1])
                emb = icd2emb.get(icd[:-2], None)
                if emb is None:
                    # print('Warning 3: ICD code not found in the embedding:', icd[:-2])
                    emb = np.zeros(emb_dim, dtype=float)
                    cannot_find += 1
        x[pos] = emb

    print('Cannot find:', cannot_find)
    return torch.from_numpy(x)


if __name__ == '__main__':
    data_path = '../data'
    dataset = 'mimic3'  # 'mimic3' or 'eicu'
    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    encoded_path = os.path.join(dataset_path, 'encoded')
    standard_path = os.path.join(dataset_path, 'standard')

    # Necessary Input: pid, patient_admission, admission_codes_encoded, code_num
    # Additional Input: admission_labs_encoded, lab_map
    pretrain_pids = pickle.load(open(os.path.join(encoded_path, 'pids.pkl'), 'rb'))['pretrain_pids']
    patient_admission = pickle.load(open(os.path.join(encoded_path, 'patient_admission.pkl'), 'rb'))
    admission_codes_encoded = pickle.load(open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'rb'))
    code_map_pretrain = pickle.load(open(os.path.join(encoded_path, 'code_maps.pkl'), 'rb'))['code_map_pretrain']
    code_num_pretrain = len(code_map_pretrain)
    admission_labs_encoded = pickle.load(open(os.path.join(encoded_path, 'labs_encoded.pkl'), 'rb'))
    lab_map = pickle.load(open(os.path.join(encoded_path, 'code_maps.pkl'), 'rb'))['lab_map']

    emb_path = os.path.join(data_path, 'emb')
    icd9cm_emb = pickle.load(open(os.path.join(emb_path, 'icd9cm_emb.pkl'), 'rb'))
    icd2name = icd9cm_emb['icd2name']
    icd2emb = icd9cm_emb['icd2emb']

    icd2hake = pickle.load(open(os.path.join(emb_path, 'ICD2HAKE.pkl'), 'rb'))

    tuples_disease2disease = generate_disease_complication_edge_index(pretrain_pids,
                                                                      patient_admission,
                                                                      admission_codes_encoded,
                                                                      code_num_pretrain)
    edge_index_disease2disease, edge_index_codes_details_disease2disease = tuples_disease2disease
    x_disease2disease = generate_disease_complication_x(icd2emb, code_map_pretrain, emb_dim=1536)
    x_disease2disease_HAKE = generate_disease_complication_x(icd2hake, code_map_pretrain, emb_dim=2000)

    print(len(edge_index_disease2disease))
    print(len(edge_index_disease2disease[0]))
    print(len(edge_index_codes_details_disease2disease))
    print(x_disease2disease.shape)
    print(x_disease2disease_HAKE.shape)
