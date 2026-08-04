# Deep Learning-Based Segmentation and Feature-Based Clinical Prediction for Head and Neck Cancer

![Figure 1](./Images/Fig.%201.png)

# Reproducing the Experiments

The following steps reproduce the complete five-fold cross-validation pipeline used in this study. The commands below illustrate the workflow for **Fold 1**. The same procedure should be repeated for the remaining folds by replacing the corresponding training and validation split files and output directories.

It is assumed that the HECKTOR 2026 dataset has already been downloaded from the official challenge website and preprocessed according to the instructions provided in this repository.

## 1. Clone the Repository

Clone the repository and navigate to its root directory.

```bash
git clone [<repository_url>](https://github.com/segment-pioneers/hecktor2026)
cd hecktor2026
```

## 2. Create a Python Environment

It is strongly recommended to create a new Python environment before installing the required packages, as **imgaug** is incompatible with recent NumPy releases.

```bash
python -m venv hecktor_env
source hecktor_env/bin/activate      # Linux/macOS
```

or

```bash
conda create -n hecktor python=3.10
conda activate hecktor
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 3. Navigate to the Source Directory

```bash
cd src
```

## 4. Configure the Python Package Path

This only needs to be executed once per terminal session.

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

## 5. Segmentation

Train the AdaMSS segmentation model for Fold 1.

```bash
python -m segmentation.train_seg \
    --data_dir /path/to/preprocessed_data/ \
    --train_samples /path/to/data_splits/fold1_train.npy \
    --valid_samples /path/to/data_splits/fold1_valid.npy \
    --model_dir /path/to/output/segmentation/fold1/ \
    --log_dir /path/to/output/segmentation/fold1/logs/ \
    --epochs 50 \
    --steps_per_epoch 200 \
    --batch_size 2 \
    --device gpu0
```

## 6. TN Stage Classification

### 6.1 Extract Handcrafted Features

Extract handcrafted radiomic, anatomical, and clinical features from the preprocessed data.

```bash
python -m tn_staging.extract_tn_features \
    --data_dir /path/to/preprocessed_data/ \
    --clinical_csv /path/to/HECKTOR_2026_training_data.csv \
    --output_cache /path/to/output/gt_tn_features_37.pkl
```

### 6.2 Train the TN Stage Classifiers

Train the independent CatBoost classifiers for T-stage and N-stage prediction.

```bash
python -m tn_staging.train_tn \
    --feature_cache /path/to/output/gt_tn_features_37.pkl \
    --clinical_csv /path/to/HECKTOR_2026_training_data.csv \
    --train_samples /path/to/data_splits/fold1_train.npy \
    --valid_samples /path/to/data_splits/fold1_valid.npy \
    --model_dir /path/to/output/tn_staging/fold1/ \
    --seed 42
```

### 6.3 Generate TN Stage Probability Predictions

Generate the TN-stage probability distributions that will be used by the survival prediction model.

```bash
python -m tn_staging.predict_tn \
    --feature_cache /path/to/output/gt_tn_features_37.pkl \
    --t_model_path /path/to/output/tn_staging/fold1/best_t_model.cbm \
    --n_model_path /path/to/output/tn_staging/fold1/best_n_model.cbm \
    --output_file /path/to/output/tn_staging/fold1/tn_predictions.pkl
```

## 7. Recurrence-Free Survival Prediction

### 7.1 Extract Survival Features

Extract the handcrafted feature representation used for recurrence-free survival prediction.

```bash
python -m survival.extract_surv_features \
    --data_dir /path/to/preprocessed_data/ \
    --clinical_csv /path/to/HECKTOR_2026_training_data.csv \
    --tn_predictions /path/to/output/tn_staging/fold1/tn_predictions.pkl \
    --seg_model /path/to/output/segmentation/fold1/best.pt \
    --device cuda \
    --train_samples /path/to/data_splits/fold1_train.npy \
    --valid_samples /path/to/data_splits/fold1_valid.npy \
    --output_file /path/to/output/survival/fold1/survival_features.pkl
```

### 7.2 Train the Survival Prediction Model

Train the Random Survival Forest (RSF).

```bash
python -m survival.train_surv \
    --features_file /path/to/output/survival/fold1/survival_features.pkl \
    --model_dir /path/to/output/survival/fold1/ \
    --random_state 42
```

## 8. Repeat for the Remaining Folds

Repeat **Sections 5–7** for Folds **2–5** by replacing the corresponding training and validation split files and output directories.

## 9. Inference

The inference pipeline used for the HECKTOR 2026 challenge submission is provided in:

```text
src/inference
```

Please refer to **`src/inference/README.md`** for instructions on reproducing the inference pipeline used in the HECKTOR 2026 challenge submission.
