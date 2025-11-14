# Medical Knowledge Graph for DuaLK

This directory contains the comprehensive medical Knowledge Graph (KG) used in the DuaLK framework. The KG integrates medical entities and relations from both **ontology-based sources** and **LLM-generated structured knowledge**, serving as the foundation for both **pretraining** and **knowledge-guided diagnostic reasoning**.

**Overall Statistics:**
- **52,604** types of nodes
- **3,645** types of entities
- **560,183** triples in total

---

## 1. Ontology KG

### Overview

The Ontology-KG captures comprehensive medical relationships including **drug-drug interactions**, **drug-disease relationships** (indication, contraindication, off-label use), **disease-phenotype associations**, and **disease-disease hierarchies**. It is constructed from multiple authoritative medical ontology databases and provides a structured knowledge backbone to enhance the model's reasoning capability in downstream diagnostic tasks.

### Data Sources

The Ontology KG integrates five primary standardized medical ontologies:

1. **DrugBank** (Knox et al. [2024])
   - **Version:** 5.1.12, published March 14, 2024
   - **Content:** Comprehensive pharmaceutical knowledge with detailed drug information
   - **Focus:** Synergistic drug interactions (bidirectional connections between drugs)

2. **DrugCentral** (Ursu et al. [2016])
   - **Version:** SQL Database released November 1, 2023
   - **Content:** Curated drug-disease interactions
   - **Focus:** Drug indications, contraindications, and off-label uses

3. **HPO (Human Phenotype Ontology)** (Gargano et al. [2024])
   - **Version:** Updated April 19, 2024
   - **Content:** Detailed phenotype abnormalities associated with diseases
   - **Focus:** Disease-phenotype and phenotype-phenotype relationships with expertly curated annotations

4. **ICD-9-CM Disease Ontology** (Organization et al. [1988])
   - **Content:** Coding system for classifying medical conditions used in EHR data
   - **Focus:** Parent-child relationships among disease codes to describe disease hierarchies

5. **SIDER (Side Effect Resource)** (Kuhn et al. [2016])
   - **Version:** Data release dated October 21, 2015
   - **Content:** Side-effect phenotypes caused by various drugs
   - **Focus:** Drug side effects and adverse reactions

### File Structure

```
Ontology-KG/
└── kg_ontology.csv          # Complete ontology-based knowledge graph
```

### File Format

**`kg_ontology.csv` Structure:**

Each row represents a directed triple: `[head_entity, relation, tail_entity]`, forming the edges of the knowledge graph.

- **Column 1 (`x_name`)**: Head entity (source node)
  - Drug entities: prefixed with `ATC_` (Anatomical Therapeutic Chemical classification code)
  - Disease entities: prefixed with `ICD_` (ICD-9-CM diagnosis code)
  - Phenotype entities: HPO terms or other phenotype identifiers

- **Column 2 (`display_relation`)**: Relationship type between entities
  - `indication`: Drug is indicated for treating a disease
  - `off_label_use`: Drug is used off-label for a disease
  - `contraindication`: Drug is contraindicated for a disease
  - `synergistic_interaction`: Synergistic interaction between two drugs
  - `disease_phenotype`: Disease is associated with a phenotype
  - `parent_child`: Hierarchical relationship between diseases (ICD hierarchy)
  - `side_effect`: Drug causes a side effect/adverse phenotype

- **Column 3 (`y_name`)**: Tail entity (target node)
  - Same entity type conventions as `x_name`

**Example triples:**
```csv
x_name,display_relation,y_name
ATC_V04CG,indication,ICD_111.7
ATC_R07AX,indication,ICD_993.4
ATC_J05AX,indication,ICD_581.2
ATC_G03BA,off_label_use,ICD_228.0
ATC_M05BA,indication,ICD_156.6
```

---

## 2. LLM KG

### Overview

The LLM-KG component transforms **unstructured clinical text** from various medical knowledge databases into **structured triples** using Large Language Models. This process extracts rich, human-readable medical knowledge and converts it into a graph-structured format suitable for computational reasoning.

### Why LLM-based Triple Extraction?

Traditional ontology-based KGs, while authoritative, often lack detailed clinical descriptions, symptom presentations, treatment guidelines, and patient management information. By leveraging LLMs to process clinical text from reputable medical sources, we:

1. **Enrich knowledge coverage** - Capture nuanced clinical information beyond simple drug-disease relationships
2. **Maintain currency** - Enable extraction from the latest medical literature and guidelines
3. **Preserve clinical context** - Extract relationships that reflect real-world diagnostic and treatment pathways

### Triple Generation Method

We use **[Llama-prompting.ipynb](LLM-KG/Llama-prompting.ipynb)** for converting raw medical text into structured triples.

**Model Choice:**
To lower the barrier to entry and ensure reproducibility, we use **Llama-2-13B-chat-GPTQ** (8-bit quantized) for inference. However, our experiments demonstrate that **all Llama family models** (including Llama-3.1-8B, Llama-2-7B, etc.) produce highly similar triple extraction results.

**Key Features of the Prompting Pipeline:**
- **Few-shot prompting** with carefully designed examples
- **Structured output format** enforcing `[ENTITY_1, RELATION, ENTITY_2]` triples
- **Domain-specific constraints** ensuring medical entity consistency
- **Topic-based extraction** processing text by clinical sections (Overview, Symptoms, Treatment, etc.)

**Example Extraction:**

Input text about "Cyclothymia":
```
Cyclothymia causes emotional ups and downs, but they're not as extreme as those
in bipolar I or II disorder. Treatment options include talk therapy, medications
and close follow-up with your doctor.
```

Generated triples:
```python
('Cyclothymia', 'IS_A', 'Mood Disorder')
('Cyclothymia', 'CAUSES', 'Emotional Ups and Downs')
('Cyclothymia', 'NEEDS_TREATMENT', 'Talk Therapy')
('Cyclothymia', 'NEEDS_TREATMENT', 'Medications')
```

### Data Sources

The LLM KG is constructed from four major clinical knowledge databases:

#### 2.1 Mayo Clinic
- **Source:** [Mayo Clinic Disease & Conditions](https://www.mayoclinic.org/diseases-conditions)
- **Content Type:** Comprehensive patient-oriented disease information
- **Coverage:** Common and complex diseases with detailed descriptions
- **Sections Extracted:** Overview, Symptoms, Causes, Risk Factors, Diagnosis, Treatment, Prevention
- **Version:** As of data collection date (to be updated with exact timestamp)
- **Output File:** `mayoclinic_triples.txt` (2.2 MB)

#### 2.2 OrphaNet
- **Source:** [OrphaNet - The portal for rare diseases and orphan drugs](https://www.orpha.net/)
- **Content Type:** Specialized rare disease knowledge base
- **Coverage:** Over 6,000 rare diseases with expert-curated information
- **Sections Extracted:** Definition, Prevalence, Epidemiology, Clinical Description, Management and Treatment
- **Version:** As of data collection date (to be updated with exact timestamp)
- **Output File:** `OrphaNet_triples.txt` (5.8 MB)
- **Mapping File:** `auxiliary/orpha2icd.pkl` (OrphaNet ID to ICD code mapping)

#### 2.3 RareDisease.info
- **Source:** [RareDisease.info](https://rarediseases.info.nih.gov/)
- **Content Type:** NIH-supported rare disease information
- **Coverage:** Rare and genetic diseases with clinical summaries
- **Sections Extracted:** Clinical features, Genetics, Diagnosis, Treatment
- **Version:** As of data collection date (to be updated with exact timestamp)
- **Output File:** `RareDisease_triples.txt` (368 KB)
- **Mapping File:** `auxiliary/rare2icd.csv` (RareDisease ID to ICD code mapping)

#### 2.4 Wikipedia Medical Articles
- **Source:** Wikipedia Medical Project
- **Content Type:** Community-curated medical knowledge with citations
- **Coverage:** Broad coverage of diseases with extra clinical knowledge
- **Sections Extracted:** Pathophysiology, Diagnosis, Treatment, Epidemiology, Extra Knowledge sections
- **Version:** As of data collection date (to be updated with exact timestamp)
- **Output File:** `wiki_triples.txt` (2.9 MB)
- **Mapping File:** `auxiliary/wiki2icd.pkl` (Wikipedia article to ICD code mapping)

### File Structure

```
LLM-KG/
├── Llama-prompting.ipynb          # LLM-based triple extraction pipeline
├── mayoclinic_triples.txt         # Extracted triples from Mayo Clinic
├── OrphaNet_triples.txt           # Extracted triples from OrphaNet
├── RareDisease_triples.txt        # Extracted triples from RareDisease.info
├── wiki_triples.txt               # Extracted triples from Wikipedia
└── auxiliary/                     # Auxiliary mapping files
    ├── mayo2icd.csv              # Mayo Clinic disease to ICD-9 mapping
    ├── orpha2icd.pkl             # OrphaNet disease to ICD-9 mapping
    ├── rare2icd.csv              # RareDisease to ICD-9 mapping
    ├── wiki2icd.pkl              # Wikipedia disease to ICD-9 mapping
    ├── mayo.pkl                  # Raw Mayo Clinic data (serialized)
    ├── orpha.pkl                 # Raw OrphaNet data (serialized)
    ├── rare.pkl                  # Raw RareDisease data (serialized)
    └── wiki.pkl                  # Raw Wikipedia data (serialized)
```

**Triple File Format:**
Each `.txt` file contains disease-specific triples organized by topics:
```
Disease Name # Topic
    (ENTITY_1, RELATION, ENTITY_2)
    (ENTITY_1, RELATION, ENTITY_2)
    ...

Next Disease Name # Topic
    ...
```

---

## Data Collection and Future Updates

### Current Status

The knowledge graphs included in this repository represent a snapshot of medical knowledge extracted from the sources listed above. Due to repository size limitations, we provide:

- **Complete Ontology KG** in `Ontology-KG/kg_ontology.csv`
- **Sample LLM-generated triples** as demonstrations in `LLM-KG/`

### Commitment to Open Source

**We commit to releasing all data collection and web scraping code upon paper acceptance.** This will enable researchers to:

1. **Extract the latest medical knowledge** from continuously updated sources
2. **Reproduce our knowledge graph construction** with current data
3. **Extend the framework** to additional medical knowledge sources
4. **Adapt the pipeline** for domain-specific clinical applications

The code release will include:
- Web scrapers for Mayo Clinic, OrphaNet, RareDisease.info, and Wikipedia
- ICD code mapping utilities
- Preprocessing and cleaning scripts
- Complete LLM prompting pipeline with optimized templates

By providing these tools, we aim to support the research community in maintaining up-to-date medical knowledge graphs that reflect the latest clinical understanding and diagnostic guidelines.

---

## Usage in DuaLK Framework

Both Ontology KG and LLM KG are integrated into the DuaLK framework for:

1. **Knowledge Graph Embedding Pretraining** - Learning entity and relation representations
2. **Diagnostic Reasoning** - Guiding the model's attention to relevant clinical pathways
3. **Interpretability** - Providing transparent knowledge-based explanations for predictions

For embedding generation, we use [HAKE](https://github.com/MIRALab-USTC/KGE-HAKE) to obtain entity embeddings. See `../data/emb/` for embedding files.

---

## Citation

If you use this knowledge graph in your research, please cite our paper:

```
@article{hu2024bridging,
  title={Bridging Stepwise Lab-Informed Pretraining and Knowledge-Guided Learning for Diagnostic Reasoning},
  author={Hu, Pengfei and Lu, Chang and Wang, Fei and Ning, Yue},
  journal={arXiv preprint arXiv:2410.19955},
  year={2024}
}
```


