# Preprocessing and Anatomical-Prior Coverage Analysis

This directory contains the preprocessing pipeline and the analysis used to evaluate the coverage of the predefined anatomical-prior crops for the HECKTOR 2026 training dataset.

## Contents

### `Preprocess_224.ipynb`

This notebook implements the preprocessing pipeline used for the experiments. The pipeline:

- resamples CT, PET, and segmentation masks to 1 mm isotropic resolution;
- classifies scans as `HEAD_NECK` or `FULL_BODY` based on their physical extent in the z-direction;
- determines a scan-specific crop center using predefined anatomical priors;
- extracts a `224³` voxel crop centered at the anatomical prior;
- pads the crop when necessary;
- normalizes the CT and PET volumes; and
- saves the preprocessed data as compressed `.npz` files.

The resulting preprocessed data are used by the segmentation, TN staging, and recurrence-free survival pipelines described in the main repository README.

### `anatomical_prior_coverage_analysis.ipynb`

This notebook evaluates whether the predefined anatomical-prior crops retain the ground-truth primary and nodal tumors.

The analysis is performed after resampling the scans and segmentation masks to 1 mm isotropic resolution. Three candidate crop sizes are evaluated:

- `160³`
- `192³`
- `224³`

For each patient and crop size, GTVp and GTVn coverage is calculated as:

> **Coverage (%) = (Retained tumor voxels / Total ground-truth tumor voxels) × 100**

The analysis is intended to quantify the extent to which the fixed anatomical-prior crop may exclude tumors in atypical locations.

## Batch-wise Processing

The HECKTOR 2026 training dataset contains 782 patients. Due to available storage limitations, the coverage analysis was performed in batches rather than processing all patients simultaneously.

Four batches were used:

- Batch 1: 200 patients
- Batch 2: 200 patients
- Batch 3: 200 patients
- Batch 4: 182 patients

Patient-level results were saved separately for each batch and subsequently combined for the final analysis across all 782 patients.

The batch identifiers are reflected in the corresponding CSV filenames.

## Coverage Analysis Outputs

### Patient-level coverage files

The following files contain patient-level GTVp and GTVn coverage measurements for each crop size:

- `crop_coverage_patient_level_batch_01.csv`
- `crop_coverage_patient_level_batch_02.csv`
- `crop_coverage_patient_level_batch_03.csv`
- `crop_coverage_patient_level_batch_04.csv`

Each row corresponds to one patient and contains:

- patient identifier;
- scan type (`HEAD_NECK` or `FULL_BODY`);
- GTVp coverage for `160³`, `192³`, and `224³` crops; and
- GTVn coverage for `160³`, `192³`, and `224³` crops.

### Batch-level summary files

The following files contain coverage-threshold summaries for each batch:

- `crop_coverage_summary_batch_01.csv`
- `crop_coverage_summary_batch_02.csv`
- `crop_coverage_summary_batch_03.csv`
- `crop_coverage_summary_batch_04.csv`

The summaries report the number and percentage of cases satisfying the following coverage thresholds:

- 100%
- ≥99%
- ≥95%
- ≥90%
- ≥75%
- ≥50%
- <50%

for both GTVp and GTVn and for all three candidate crop sizes.

### Combined analysis

`crop_coverage_final_782_patients.csv` contains the final coverage-threshold analysis after combining the patient-level results from all four batches.

The combined dataset contains all 782 patients and is checked for duplicate patient identifiers before the final statistics are calculated.

### Paper-friendly table

`crop_coverage_paper_table.csv` contains the compact version of the coverage analysis used for reporting in the manuscript.

It summarizes the number and percentage of patients satisfying each coverage threshold for GTVp and GTVn.

## Reproducing the Coverage Analysis

To reproduce the analysis:

1. Obtain the HECKTOR 2026 training dataset from the official challenge website.
2. Place the required raw CT and segmentation files under the directory specified by `ROOT` in `anatomical_prior_coverage_analysis.ipynb`.
3. Run the notebook for each batch of patients.
4. Save the resulting patient-level and batch-level CSV files.
5. Combine the batch-level patient results using the final-analysis cells in the notebook.
6. Run the final analysis cells to reproduce the coverage statistics and the paper-friendly table.

The notebook contains the complete implementation of the coverage calculation and aggregation procedure.

## Notes

- Coverage is calculated on the ground-truth segmentation masks after resampling to 1 mm isotropic resolution.
- GTVp and GTVn are identified using the corresponding labels in the HECKTOR segmentation masks.
- The analysis evaluates crop coverage only; it does not retrain the segmentation, TN staging, or survival models for the different crop sizes.
- The `224³` crop corresponds to the crop size used in the final preprocessing pipeline.
