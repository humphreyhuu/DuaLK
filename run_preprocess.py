import os
import _pickle as pickle

from sklearn.utils import shuffle

from preprocess.parse_csv import Mimic3Parser, process_lab_events, process_lab_items  # Mimic4Parser,
from preprocess.encode import encode_code, encode_lab
from preprocess.build_dataset import split_patients, build_code_xy, build_code_xy_pretrain, build_single_lab_xy
from preprocess.build_dataset import build_heart_failure_y, get_rare_codes  # build_hierarchical_y,
# For Embedding Initialization
from preprocess.auxiliary import load_icd2name
# For Graph Construction
from preprocess.generate_graph import generate_disease_complication_edge_index, generate_disease_complication_x


if __name__ == '__main__':
    conf = {
        'mimic3': {
            'parser': Mimic3Parser,
            'train_num': 6000,
            'test_num': 1000,
            'threshold': 0.01
        },
        # 'mimic4': {
        #     'parser': Mimic4Parser,
        #     'train_num': 8000,
        #     'test_num': 1000,
        #     'threshold': 0.01,
        #     'sample_num': 10000
        # },
    }
    data_path = 'data'
    dataset = 'mimic3'  # 'mimic3', 'mimic4'
    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the CSV files in `data/%s/raw`' % dataset)
        exit()
    parser = conf[dataset]['parser'](raw_path)
    patient_admission, admission_codes = parser.parse()

    print('There are %d valid patients' % len(patient_admission))
    max_admission_num = max([len(admissions) for admissions in patient_admission.values()])
    max_code_num_in_a_visit = max([len(codes) for codes in admission_codes.values()])
    print('max admission num: %d, max code num in an admission: %d' % (max_admission_num, max_code_num_in_a_visit))

    # Parsing Lab Results
    admission_labs, lab_set = process_lab_events(raw_path, admission_codes, dataset)
    lab_category = process_lab_items(raw_path, lab_set, dataset)
    max_lab_num_in_a_visit = max([len(labs) for labs in admission_labs.values()])
    print('max lab num in an admission: %d' % max_lab_num_in_a_visit)

    # Encoding
    admission_codes_encoded, code_map, code_map_pretrain = encode_code(patient_admission, admission_codes)
    # Additional keys:'all','hematology','chemistry', 'blood gas'
    admission_labs_encoded, lab_map = encode_lab(admission_labs, lab_category)
    lab_num = len(lab_map['all'])
    print('There are %d unique labs' % lab_num)

    resource_path = 'resources'  # Path for ICD9CM.csv file
    icd2name, _ = load_icd2name(resource_path, code_map_pretrain)  # involving codes within single & multiple visits
    print('There are %d matched ICD9CM codes' % len(icd2name))

    code_num = len(code_map)  # 4880
    code_num_pretrain = len(code_map_pretrain)  # 6981 - 6427 only appear in single visit data
    print('There are %d pretrain codes, %d codes in multiple visits' % (code_num_pretrain, code_num))

    single_pids, pretrain_pids, train_pids, valid_pids, test_pids = split_patients(  # Type: numpy array
        patient_admission=patient_admission,
        admission_codes=admission_codes,
        code_map=code_map,
        train_num=conf[dataset]['train_num'],
        test_num=conf[dataset]['test_num'], seed=42  # 6669
    )
    print('There are %d single samples' % len(single_pids))
    print('There are %d pretrain, %d train, %d valid, %d test samples' %
          (len(pretrain_pids), len(train_pids), len(valid_pids), len(test_pids)))

    # code_levels_pretrain = generate_code_levels(data_path, code_map_pretrain)  # 2D array: (6982, 4)
    # code_levels = code_levels_pretrain[:(code_num + 1)]  # 2D array: (4881, 4)
    # # Level 1: 19; Level 2: 181, Level 3: 1069 [index staring from 1 for each level]
    # # SubClass is a List of List involving array of children indices
    # subclass_maps_pretrain = generate_subclass_map(code_level_matrix=code_levels_pretrain)
    # subclass_maps = generate_subclass_map(code_level_matrix=code_levels)
    # # Graph only considering disease-disease interactions
    # code_code_adj = generate_code_code_adjacent(pids=pretrain_pids, patient_admission=patient_admission,
    #                                             admission_codes_encoded=admission_codes_encoded,
    #                                             code_num=code_num_pretrain)  # 2D array: (6982, 6982)

    # Disease-Complication Graph Construction
    tuples_disease2disease = generate_disease_complication_edge_index(pretrain_pids, patient_admission,
                                                                      admission_codes_encoded, code_num_pretrain,
                                                                      self_edge=True, edge_score=1)
    edge_index_disease2disease, edge_weight_disease2disease = tuples_disease2disease
    print('There are %d disease-disease edges' % len(edge_index_disease2disease[0]))
    # Disease Nodes Embedding Initialization
    emb_path = os.path.join(data_path, 'emb')
    # icd9cm_emb = pickle.load(open(os.path.join(emb_path, 'icd9cm_emb.pkl'), 'rb'))
    # icd2emb = icd9cm_emb['icd2emb']
    # x_disease2disease = generate_disease_complication_x(icd2emb, code_map_pretrain)

    kg_embed_dims = [2000]
    x_disease2disease_HAKE = {}
    for kg_embed_dim in kg_embed_dims:
        print('Initializing HAKE %d embeddings for graph ...' % kg_embed_dim)
        icd2hake = pickle.load(open(os.path.join(emb_path, f'ICD2HAKE_{kg_embed_dim}.pkl'), 'rb'))
        x_disease2disease_HAKE[kg_embed_dim] = generate_disease_complication_x(icd2hake, code_map_pretrain,
                                                                               emb_dim=kg_embed_dim)

    # Only for single visits data - VARIABLE: USE_LAB
    lab_x, lab_y = build_single_lab_xy(single_pids, patient_admission,
                                       admission_codes_encoded, admission_labs_encoded['all'])
    lab_x, lab_y = shuffle(lab_x, lab_y, random_state=66)
    single_train_labs_x, single_train_labs_y = lab_x[:int(0.85 * lab_x.shape[0])], lab_y[:int(0.85 * lab_y.shape[0])]
    single_test_labs_x, single_test_labs_y = lab_x[int(0.85 * lab_x.shape[0]):], lab_y[int(0.85 * lab_y.shape[0]):]

    # Data Preparation for Pretraining LAB
    pretrain_codes = build_code_xy_pretrain(pretrain_pids, patient_admission, admission_codes_encoded,
                                            admission_labs_encoded, max_admission_num, lab_map,
                                            max_code_num_in_a_visit, lab_category='all')
    pretrain_codes_x, pretrain_codes_y_all, pretrain_visit_lens = pretrain_codes
    _, pretrain_codes_y_hema, _ = build_code_xy_pretrain(pretrain_pids, patient_admission, admission_codes_encoded,
                                                         admission_labs_encoded, max_admission_num, lab_map,
                                                         max_code_num_in_a_visit, lab_category='hematology')
    _, pretrain_codes_y_chem, _ = build_code_xy_pretrain(pretrain_pids, patient_admission, admission_codes_encoded,
                                                         admission_labs_encoded, max_admission_num, lab_map,
                                                         max_code_num_in_a_visit, lab_category='chemistry')
    _, pretrain_codes_y_blood, _ = build_code_xy_pretrain(pretrain_pids, patient_admission, admission_codes_encoded,
                                                          admission_labs_encoded, max_admission_num, lab_map,
                                                          max_code_num_in_a_visit, lab_category='blood gas')
    pretrain_codes_y = pretrain_codes_y_all, pretrain_codes_y_hema, pretrain_codes_y_chem, pretrain_codes_y_blood

    # Data Preparation for Training, Validation, and Testing
    train_codes_x, train_codes_y, train_visit_lens, train_labs_x = build_code_xy(train_pids, patient_admission,
                                                                                 admission_labs_encoded['all'], lab_num,
                                                                                 admission_codes_encoded,
                                                                                 max_admission_num,
                                                                                 code_num, max_code_num_in_a_visit)
    valid_codes_x, valid_codes_y, valid_visit_lens, valid_labs_x = build_code_xy(valid_pids, patient_admission,
                                                                                 admission_labs_encoded['all'], lab_num,
                                                                                 admission_codes_encoded,
                                                                                 max_admission_num,
                                                                                 code_num, max_code_num_in_a_visit)
    test_codes_x, test_codes_y, test_visit_lens, test_labs_x = build_code_xy(test_pids, patient_admission,
                                                                             admission_labs_encoded['all'], lab_num,
                                                                             admission_codes_encoded,
                                                                             max_admission_num,
                                                                             code_num, max_code_num_in_a_visit)

    print('Subsampling rare diseases ...')
    codes_y_rare = get_rare_codes(train_codes_y, valid_codes_y, test_codes_y, threshold=3)
    train_codes_y_r, valid_codes_y_r, test_codes_y_r = codes_y_rare
    print(train_codes_y.shape, valid_codes_y.shape, test_codes_y.shape)  # 4880
    print(train_codes_y_r.shape, valid_codes_y_r.shape, test_codes_y_r.shape)  # 3096

    print(f'lab_x shape: {lab_x.shape}')
    print(f'lab_y shape: {lab_y.shape}')
    print(single_train_labs_x.shape, single_train_labs_y.shape)
    print(f'train_labs_x shape: {train_labs_x.shape}')

    train_hf_y = build_heart_failure_y('428', train_codes_y, code_map)
    valid_hf_y = build_heart_failure_y('428', valid_codes_y, code_map)
    test_hf_y = build_heart_failure_y('428', test_codes_y, code_map)

    train_labs_data = (single_train_labs_x, single_train_labs_y)
    test_labs_data = (single_test_labs_x, single_test_labs_y)
    pretrain_codes_data = (pretrain_codes_x, pretrain_codes_y, None, pretrain_visit_lens)  # pretrain_y_h
    train_codes_data = (train_codes_x, train_codes_y, train_labs_x,
                        train_codes_y_r)  # train_y_h, train_visit_lens
    valid_codes_data = (valid_codes_x, valid_codes_y, valid_labs_x,
                        valid_codes_y_r)  # valid_y_h, valid_visit_lens
    test_codes_data = (test_codes_x, test_codes_y, test_labs_x,
                       test_codes_y_r)  # test_y_h, test_visit_lens

    encoded_path = os.path.join(dataset_path, 'encoded')
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)

    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
    pickle.dump(admission_labs_encoded, open(os.path.join(encoded_path, 'labs_encoded.pkl'), 'wb'))
    pickle.dump({
        'lab_map': lab_map,
        'code_map': code_map,
        'code_map_pretrain': code_map_pretrain
    }, open(os.path.join(encoded_path, 'code_maps.pkl'), 'wb'))
    pickle.dump({
        'pretrain_pids': pretrain_pids,
        'train_pids': train_pids,
        'valid_pids': valid_pids,
        'test_pids': test_pids
    }, open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))

    print('saving standard data ...')
    standard_path = os.path.join(dataset_path, 'standard')
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    pickle.dump(pretrain_codes_data, open(os.path.join(standard_path, 'pretrain_codes_dataset.pkl'), 'wb'))
    pickle.dump({
        'train_labs_data': train_labs_data,
        'test_labs_data': test_labs_data
    }, open(os.path.join(standard_path, 'labs_dataset.pkl'), 'wb'))
    pickle.dump({
        'train_codes_data': train_codes_data,
        'valid_codes_data': valid_codes_data,
        'test_codes_data': test_codes_data
    }, open(os.path.join(standard_path, 'codes_dataset.pkl'), 'wb'))
    pickle.dump({
        'train_hf_y': train_hf_y,
        'valid_hf_y': valid_hf_y,
        'test_hf_y': test_hf_y
    }, open(os.path.join(standard_path, 'heart_failure.pkl'), 'wb'))
    # pickle.dump({
    #     'code_levels': code_levels,
    #     'code_levels_pretrain': code_levels_pretrain,
    #     'subclass_maps': subclass_maps,
    #     'subclass_maps_pretrain': subclass_maps_pretrain,
    #     'code_code_adj': code_code_adj
    # }, open(os.path.join(standard_path, 'auxiliary.pkl'), 'wb'))

    graph_path = os.path.join(dataset_path, 'graph')
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    pickle.dump({
        'edge_index': edge_index_disease2disease,
        'edge_weight': edge_weight_disease2disease,
        # 'x': x_disease2disease,
        # 'x_hake_256': x_disease2disease_HAKE[256],
        # 'x_hake_512': x_disease2disease_HAKE[512],
        'x_hake_2000': x_disease2disease_HAKE[2000],
    }, open(os.path.join(graph_path, 'disease2disease.pkl'), 'wb'))
