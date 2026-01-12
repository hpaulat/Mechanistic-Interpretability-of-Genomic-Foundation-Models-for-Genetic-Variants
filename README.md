## Overview

This repository analyzes ClinVar genetic variants using a nucleotide transformer model. For each variant, fixed-length genomic windows are extracted around the mutation, reference (REF) and alternative (ALT) sequences are embedded, and mutation-induced differences are analyzed downstream (e.g. classification, attention statistics).

## External dependency: genomic-FM (submodule)

This project depends on the genomic-FM repository, which is included as a git submodule under:

```
external/genomic-FM/
```

After cloning this repository, make sure to initialize and update the submodule:

```
git submodule update --init --recursive
```

All data-loading utilities (including ClinVar processing) are provided by this submodule.

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

All scripts assume the file exists at this exact location.
