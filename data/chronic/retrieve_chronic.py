import pandas as pd
import pickle


def parse_chronic_codes(data_path, chronic_name):
    # Load the data from the specified Excel file path
    data = pd.read_excel(data_path)
    chronic_icd, chronic_pos = [], []

    for index, row in data.iterrows():
        if row[chronic_name] == 'YES':
            chronic_icd.append(row['Code'])
            chronic_pos.append(row['ID'])
        elif pd.notna(row[chronic_name]):
            print(f"Unexpected value in row {index}: {row[chronic_name]}")

    # Return the lists of chronic codes and their positions
    return chronic_icd, chronic_pos


if __name__ == '__main__':
    chronic_icd, chronic_pos = parse_chronic_codes('../../resources/codemap_chronic.xlsx',
                                                   'Chronic?')
    print(len(chronic_icd), len(chronic_pos))

    with open('chronic_icd.pkl', 'wb') as f:
        pickle.dump(chronic_icd, f)

    with open('chronic_pos.pkl', 'wb') as f:
        pickle.dump(chronic_pos, f)
