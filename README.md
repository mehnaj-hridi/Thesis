# Batch-Wise Ensemble Model using Classical and Quantum SVM for Email Phishing Detection

## Project Overview

This thesis project implements a **hybrid classical-quantum machine learning approach** to detect email phishing attacks. The work compares the performance of classical Support Vector Machines (SVM), Quantum Support Vector Machines (QSVM), and ensemble models on phishing detection tasks.


---

## Problem Statement

Email phishing is a critical cybersecurity threat. Traditional machine learning approaches have limitations in capturing complex feature relationships. This project investigates whether quantum machine learning can provide superior performance in detecting phishing emails through:

- Advanced feature engineering from email and web metadata
- Hybrid classical-quantum model comparison
- Batch-wise quantum processing with calibration techniques
- Ensemble methods combining multiple approaches

---

## Dataset

### Source
- **File**: `email_phishing_dataset_FINAL.csv`
- **Size**: 8,000 samples, 38 Features
- **Split**: 80% training, 20% testing

### Features

The dataset includes **14 engineered features** capturing email and web characteristics.

---

## Project Structure

### Notebooks

#### 1. `FINAL_dataset_creation.ipynb`
**Purpose**: Feature engineering

**Key Processes**:
- Data loading
- Define Utility functions 
- Add 24 new engineered features
- Final dataset export

**Output**: Cleaned, engineered dataset ready for ML

---

#### 2. `FINAL_ml_qml.ipynb`
**Purpose**: Data Preprocessing, Model development and evaluation (Classical SVM, QSVM, Ensemble)

**Key Sections**:

##### A. Data Preprocessing
- Missing value handling 
- Non-numerical encoding
- MinMax scaling (0-1 normalization)
- Train-test split (80-20 stratified)
- Class distribution analysis

##### B. Feature Selection
- **Method**: Mutual Information Classification
- **Selection**: Top 6 features by MI score
- **Output**: `mi_feature_importance.png` - visualization of feature importance

##### C. Classical SVM
- Scikit-learn SVC with RBF kernel
- Batch-wise training (batch size: 500)
- Soft probability calibration
- **Outputs**:
  - `svm_final_pred.npy` - hard predictions
  - `svm_mean_proba.npy` - mean probability per sample
  - `svm_proba_matrix.npy` - probability matrix per batch
  - `svm_confusion_matrix.png`

##### D. Quantum SVM (QSVM)
- **Quantum Framework**: Qiskit 
- **Feature Map**: ZZFeatureMap with linear entanglement
- **Quantum Kernel**: FidelityQuantumKernel
- **Batch Processing**: 500 samples per batch for scalability
- **Calibration**: Sigmoid calibration on decision function
- **Outputs**:
  - `qsvm_final_pred.npy` - hard predictions
  - `qsvm_mean_proba.npy` - calibrated probabilities
  - `qsvm_proba_matrix.npy` - probability matrix per batch
  - `qsvm_confusion_matrix.png`

##### E. Ensemble Model
- **Strategy**: Averaging predictions from SVM and QSVM
- **Outputs**:
  - `ensemble_final_pred.npy` - ensemble predictions
  - `ensemble_proba.npy` - ensemble probabilities
  - `ensemble_confusion_matrix.png`

##### F. Performance Evaluation
- ROC-AUC curves for all models
- Precision-Recall-F1 comparison
- Confusion matrices
- Batch stability analysis (SVM vs QSVM across batches)
- **Outputs**: PNG visualizations for thesis figures

---

### Data Files

| File | Purpose |
|------|---------|
| `email_phishing_dataset_FINAL.csv` | Final dataset with engineered features |
| `new_dataset_classical.csv` | Original raw dataset |
| `predictions_all.csv` | Combined predictions from all models |
| `y_test.npy` | Test set ground truth labels |

### Model Outputs

| File | Description |
|------|-------------|
| `svm_final_pred.npy` | SVM hard predictions on test set |
| `svm_mean_proba.npy` | SVM soft probabilities |
| `svm_proba_matrix.npy` | SVM batch-wise probability matrix |
| `qsvm_final_pred.npy` | QSVM hard predictions on test set |
| `qsvm_mean_proba.npy` | QSVM calibrated probabilities |
| `qsvm_proba_matrix.npy` | QSVM batch-wise probability matrix |
| `ensemble_final_pred.npy` | Ensemble predictions |
| `ensemble_proba.npy` | Ensemble soft probabilities |

### Visualizations

| File | Description |
|------|-------------|
| `mi_feature_importance.png` | Top 15 features by mutual information |
| `svm_confusion_matrix.png` | SVM test set confusion matrix |
| `qsvm_confusion_matrix.png` | QSVM test set confusion matrix |
| `ensemble_confusion_matrix.png` | Ensemble test set confusion matrix |
| `roc_auc_all_models.png` | ROC-AUC curves comparison |
| `prf_comparison_all.png` | Precision-Recall-F1 comparison |
| `batch_stability_svm_vs_qsvm.png` | Batch-wise stability analysis |
| `ablation_study_results.png` | Ablation study results |

### Documentation

| File | Content |
|------|---------|
| `B1_THESIS_BOOK_final.pdf` | Full thesis manuscript |
| `thesis_hybrid_dataset_explain.pdf` | Dataset and feature engineering details |

---

## Methodology

### 1. Feature Engineering Pipeline
```
Raw Email Data
    ↓
14 Engineered Features 
    ↓
Preprocessing 
    ↓
Data Encoding → Feature Selection (MI-based, k=6)
```

### 2. Machine Learning Pipeline
```
Preprocessed Data
    ├─→ Classical SVM
    │       ├─ RBF kernel
    │       ├─ Batch-wise training
    │       └─ Sigmoid calibration
    │
    ├─→ Quantum SVM (QSVM)
    │       ├─ ZZFeatureMap (1 rep, linear entanglement)
    │       ├─ FidelityQuantumKernel
    │       ├─ Batch-wise quantum training
    │       └─ Sigmoid calibration
    │
    └─→ Ensemble (Average)
            └─ Combine SVM + QSVM probabilities
```

### 3. Quantum Feature Map
- **Circuit Type**: ZZFeatureMap
- **Qubits**: 6 (matching selected features)
- **Entanglement**: Linear (sequential qubit pairs)
- **Repetitions**: 1

---

## Key Technologies

### Classical ML
- **Framework**: Scikit-learn
- **Algorithm**: Support Vector Machines (SVM)
- **Preprocessing**: MinMax scaling, feature selection

### Quantum ML
- **Framework**: Qiskit 
- **Primitives**: Sampler
- **Kernel**: FidelityQuantumKernel with feature maps

### Data Science
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Metrics and utilities

---

## Experimental Design

### Train/Test Split
- **Train**: 80%
- **Test**:  20%
- **Stratification**: Preserves class distribution

### Batch Processing
- **Batch Size**: 500 samples
- **Reason**: Manages quantum circuit complexity and resource constraints
- **Total Batches**: 13 for training

### Evaluation Metrics
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## Results Summary

### Performance Comparison
The notebooks generate performance comparison visualizations showing:
- ROC-AUC curves for SVM, QSVM, and Ensemble
- Precision-Recall-F1 metrics across all models
- Confusion matrices for each model

### Batch Stability
- Analysis of model predictions across batches
- Variance in QSVM vs SVM performance
- Identifies quantum noise effects

### Feature Importance
- Mutual information scores for all features
- Top 6 selected features identified
- Cutoff visualization for feature selection

---

## How to Use

### Prerequisites
```bash
pip install qiskit qiskit-machine-learning
pip install scikit-learn pandas numpy matplotlib plotly
```

### Running the Notebooks

#### Step 1: Dataset Creation
```bash
jupyter notebook FINAL_dataset_creation.ipynb
```
- Loads raw dataset
- Engineers features
- Exports `email_phishing_dataset_FINAL.csv`

#### Step 2: Model Training & Evaluation
```bash
jupyter notebook FINAL_ml_qml.ipynb
```
- Loads `email_phishing_dataset_FINAL.csv` dataset
- Trains SVM, QSVM, and Ensemble models
- Generates predictions (`.npy` files)
- Creates visualizations (`.png` files)
- Outputs `predictions_all.csv` with all model predictions


---

## Directory Structure
```
g:/Thesis/
├── FINAL_dataset_creation.ipynb          # Generating engineered features
├── FINAL_ml_qml.ipynb                    # Train ML/QML models
├── email_phishing_dataset_FINAL.csv       # FINAL dataset
├── new_dataset_classical.csv              # Original Raw dataset
├── predictions_all.csv                    # Combined predictions
├── y_test.npy                             # Test labels
├── svm_final_pred.npy                     # SVM predictions
├── svm_mean_proba.npy                     # SVM probabilities
├── svm_proba_matrix.npy                   # SVM probability matrix
├── qsvm_final_pred.npy                    # QSVM predictions
├── qsvm_mean_proba.npy                    # QSVM probabilities
├── qsvm_proba_matrix.npy                  # QSVM probability matrix
├── ensemble_final_pred.npy                # Ensemble predictions
├── ensemble_proba.npy                     # Ensemble probabilities
├── mi_feature_importance.png              # Feature importance plot
├── svm_confusion_matrix.png               # SVM confusion matrix
├── qsvm_confusion_matrix.png              # QSVM confusion matrix
├── ensemble_confusion_matrix.png          # Ensemble confusion matrix
├── roc_auc_all_models.png                 # ROC-AUC comparison
├── prf_comparison_all.png                 # P-R-F1 comparison
├── batch_stability_svm_vs_qsvm.png        # Batch stability analysis
├── ablation_study_results.png             # Ablation study
├── B1_THESIS_BOOK_final.pdf               # Full thesis
├── thesis_hybrid_dataset_explain.pdf      # Dataset documentation
└── README.md                              # This file
```

---


## References & Thesis Details

- **Full Manuscript**: See `B1_THESIS_BOOK_final.pdf`
- **Dataset Details**: See `thesis_hybrid_dataset_explain.pdf`
- **Quantum ML Framework**: Qiskit Documentation (https://qiskit.org)
- **Email Phishing Detection**: CEAS, SPAM conferences publications

---

## Author & License

**Thesis**: Batch-Wise Ensemble Model using Classical and Quantum
SVM for Email Phishing Detection

**Created**: 2025-2026


---


## Quick Start Command

Run all analysis end-to-end:
```bash
# 1. Prepare dataset
jupyter nbconvert --to notebook --execute FINAL_dataset_creation.ipynb

# 2. Train models and evaluate
jupyter nbconvert --to notebook --execute FINAL_ml_qml.ipynb

# 3. Check outputs
ls -la *.npy *.png *.csv
```

---

**Last Updated**: June 2026 
**Status**: Thesis Complete
