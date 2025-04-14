# DualMAR
**Title:** DualMAR: Medical-Augmented Representation from Dual-Expertise Perspectives

Pytorch implementation of DualMAR paper

## Technical Appendix

Please see `dualmar_technical_appendix.pdf` to check detailed information.

## Requirements
- python>=3.10
- torch==2.3.1
- PyG
- CUDA==12.4
- numpy, pandas, sklearn, matplotlib

## KG Demo

KG generation results are partly shown in `kg_` folder. Because of the folder size limitation, we only shows simplified `Ontology-KG` in `./Ontology-KG/kg_ontology.csv` and some generated triples as demo shown in `LLM-KG`.

Note that, we will share the completed Diagnosis KG after the acceptance.

## KG Embedding Demo

We refer to the code of [HAKE](https://github.com/MIRALab-USTC/KGE-HAKE). Thanks for their contributions.

Furthermore, we put a simplified embeddings, which stores embeddings as `float16` with ICD9 codes appearing in MIMIC, into the `data/emb/ICD2HAKE_2000.pkl` file as shown demo to let all scripts run easily.

Note that, we will share the completed code embeddings after the acceptance.

## Run

The parameters of this current version are configured in each python file.

Note that, to transferring model into MIMIC-IV, you need to change `parse_csv_4.py` as the parsing script.

All MIMIC-III and MIMIC-IV data can be downloaded from [Physionet](https://physionet.org/content/mimiciii/1.4/), please put all data into the corresponding data path.

```bash
# Current setting is for MIMIC-III
python run_preprocess.py

# Plug-in-and-play for adding lab features in real-time diagnosis
python train_lab.py

# Proxy-task Pretraining Module
python pretrain_lab.py

# For Diagnosis Prediction
python train.py

# For Heart Failure Prediction
python train_hf.py
```

We put all hyperparameters prior to the training process for all python file, which can be tuned easily to optimize performance.
