from preprocess.get_emb_openai import embedding_retriever

import numpy as np
from tqdm import tqdm
import pickle
import os
import pandas as pd

if __name__ == '__main__':
    icd2emb = dict()
    total_tokens = 0

    data_path = 'data'
    dataset = 'mimic3'  # 'mimic3' or 'eicu'
    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    encoded_path = os.path.join(dataset_path, 'encoded')

    icd9cm = pd.read_csv(os.path.join('resources', 'ICD9CM.csv'),
                         usecols=['code', 'name'],
                         dtype={'code': str, 'name': str})
    icd2name = icd9cm.set_index('code').to_dict()['name']
    print('Total number of ICD9CM codes: ', len(icd2name))

    for icd, name in tqdm(icd2name.items()):
        embedding, num_tokens = embedding_retriever(term=name)
        icd2emb[icd] = np.array(embedding)
        total_tokens += num_tokens

    print('Total number of tokens for Embedding is: ', total_tokens)

    emb_path = os.path.join(data_path, 'emb')
    if not os.path.exists(emb_path):
        os.makedirs(emb_path)

    pickle.dump({
        'icd2name': icd2name,
        'icd2emb': icd2emb,
    }, open(os.path.join(emb_path, 'icd9cm_emb.pkl'), 'wb'))
