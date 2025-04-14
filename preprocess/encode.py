import copy
from collections import OrderedDict
from typing import Dict


def encode_code(patient_admission: dict, admission_codes: dict) -> (dict, dict, dict):
    print('encoding code ...')
    code_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        if len(admissions) <= 1:
            continue
        for admission in admissions:
            codes = admission_codes[admission['admission_id']]  # 'admission_id'
            for code in codes:
                if code not in code_map:
                    code_map[code] = len(code_map) + 1
    code_map_pretrain = copy.deepcopy(code_map)
    for pid, admissions in patient_admission.items():
        if len(admissions) > 1:
            continue
        for admission in admissions:
            codes = admission_codes[admission['admission_id']]  # 'admission_id'
            for code in codes:
                if code not in code_map_pretrain:
                    code_map_pretrain[code] = len(code_map_pretrain) + 1

    admission_codes_encoded = {
        admission_id: [code_map_pretrain[code] for code in codes]
        for admission_id, codes in admission_codes.items()
    }
    return admission_codes_encoded, code_map, code_map_pretrain


def encode_lab(admission_labs: Dict, lab_category: Dict) -> (Dict, Dict):
    """Encode lab results for admissions based on lab categories.
    :param admission_labs: Dictionary where keys are admission IDs and values are lists of lab codes.
    :param lab_category: Dictionary mapping lab codes to their respective categories.
    :return: Tuple of dictionaries with encoded lab results and lab mapping details.
    """
    admission_labs_encoded = {'hematology': {}, 'chemistry': {}, 'blood gas': {}, 'all': {}}
    lab_map = {'hematology': {}, 'chemistry': {}, 'blood gas': {}, 'all': {}}
    index_counters = {'hematology': 1, 'chemistry': 1, 'blood gas': 1, 'all': 1}
    type_counts = {'hematology': 0, 'chemistry': 0, 'blood gas': 0}  # Initialize counts for lab types

    for admission_id in admission_labs:
        for category in admission_labs_encoded:
            admission_labs_encoded[category][admission_id] = []

    for admission_id, labs in admission_labs.items():
        for lab_code in labs:
            lab_type = lab_category.get(lab_code, None)
            if lab_type in ['hematology', 'chemistry', 'blood gas']:
                type_counts[lab_type] += 1
                if lab_code not in lab_map[lab_type]:
                    lab_map[lab_type][lab_code] = index_counters[lab_type]
                    index_counters[lab_type] += 1
                admission_labs_encoded[lab_type][admission_id].append(lab_map[lab_type][lab_code])
                if lab_code not in lab_map['all']:
                    lab_map['all'][lab_code] = index_counters['all']
                    index_counters['all'] += 1
                admission_labs_encoded['all'][admission_id].append(lab_map['all'][lab_code])

    print("Lab counts per category:", type_counts)

    return admission_labs_encoded, lab_map


if __name__ == '__main__':
    # Example usage
    admission_labs = {
        '001': ['LAB1', 'LAB2', 'LAB3'],
        '002': ['LAB2', 'LAB4'],
        '003': []  # Example admission with no lab results
    }
    lab_category = {
        'LAB1': 'hematology',
        'LAB2': 'chemistry',
        'LAB3': 'blood gas',
        'LAB4': 'chemistry'
    }

    encoded_labs, lab_mappings = encode_lab(admission_labs, lab_category)
    print("Encoded Labs:", encoded_labs)
    print("Lab Mappings:", lab_mappings)
