## Overview

This repository investigates how **genomic foundation models encode the functional impact of genetic variants**, using **ClinVar single-nucleotide variants (SNVs)**. For each variant, fixed-length genomic windows are extracted around the mutation site, embedded using a **nucleotide transformer**, and analyzed by contrasting **reference (REF)** and **alternative (ALT)** alleles. Downstream analyses focus on identifying **mutation-induced representation and attention shifts** that correlate with clinical pathogenicity labels.

---

## Scientific Motivation

Large genomic foundation models achieve strong performance across tasks, yet their **internal representations remain poorly understood**, particularly at the level of single-nucleotide perturbations. Clinically annotated variants from ClinVar provide a natural setting to study:

- How transformers respond to minimal sequence edits (1 bp changes)
- Whether pathogenic variants induce systematic embedding or attention shifts
- Which sequence contexts are consistently emphasized across layers

This project serves as a **mechanistic probe**.

---

## Pipeline Overview

1. **Variant selection**

   - ClinVar SNVs with verified clinical labels (pathogenic / benign)

2. **Sequence extraction**

   - Fixed-length genomic windows centered on the variant
   - Both REF and ALT alleles generated per variant

3. **Embedding**

   - Sequences passed through a pretrained nucleotide transformer
   - Hidden states and attention weights extracted across layers

4. **Delta analysis**

   - REF vs ALT differences computed in:
     - embeddings
     - attention maps
     - downstream classifier outputs

5. **Downstream analyses**
   - Linear probing / logistic regression
   - Attention shift visualization
   - Layer-wise and position-wise attribution

---

## External dependency: genomic-FM (submodule)

This project depends on the genomic-FM repository, which is included as a git submodule under:

```
external/genomic-FM/

```

When cloning, make sure to initialize and update the submodule:

````
git clone --recurse-submodules https://github.com/hpaulat/Mechanistic-Interpretability-of-Genomic-Foundation-Models-for-Genetic-Variants.git```

````

All data-loading utilities (including ClinVar processing) are provided by this submodule.

---

## Data: Real ClinVar (verified_real_clinvar.csv)

The ClinVar dataset is not committed to the repository and must be downloaded manually.
Download the file directly into the expected data directory inside the submodule:

```
cd external/genomic-FM/root/data
curl -L -o clinvar_20240416.vcf.gz \
 "https://zenodo.org/records/11502840/files/verified_real_clinvar.csv?download=1""

```

The final path should be:

```
external/genomic-FM/root/data/verified_real_clinvar.csv

```

All scripts assume the file exists at this exact location. Larger files can easily be analyzed by modifying the scripts.

Experiments can be easily reproduced by installing dependincies:

```
pip install requirements.txt
```

and simply running the scripts. Notebooks are left to show exploration process.

## Outputs and Artifacts

Depending on the script, the repository produces:

- Variant-level embeddings (REF / ALT)
- Attention maps and attention-difference heatmaps
- Classification metrics (ROC-AUC, accuracy)
- Layer-wise attribution analyses

Outputs are saved locally and are not versioned due to size.

---

## Reproducibility Notes

- Python ≥ 3.10 recommended
- GPU strongly recommended for embeddings extractions
