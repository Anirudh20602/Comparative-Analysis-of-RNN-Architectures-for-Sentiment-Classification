# Comparative Analysis of RNN Architectures for Sentiment Classification

This repository contains the implementation, results, and report for **Homework 3: Comparative Analysis of RNN Architectures for Sentiment Classification**.  
The objective is to systematically compare RNN, LSTM, and Bidirectional LSTM models on the IMDb 50k movie review dataset, analyzing the impact of various hyperparameters.

---

## Repository Structure

```
Comparative-Analysis-of-RNN-Architectures-for-Sentiment-Classification/
├── plots/                                                                                # accuracy_vs_seq.png, f1_vs_seq.png, best_model_loss.png, worst_model_loss.png
├── comparative-analysis-of-rnn-architectures.ipynb                                       # Jupyter notebook with full implementation
├── Comparative_Analysis_of_RNN_Architectures_for_Sentiment_Classification_Report.pdf     # Final report
├── metrics.csv                                                                           # Summary of experimental results (Accuracy, F1, Training Time)
├── README.md                                                                             # Project documentation
└── requirements.txt                                                                      # Python dependencies
```

---

## Project Overview

The IMDb dataset contains 50,000 labeled movie reviews (25k train / 25k test).  
Each model is evaluated under variations in:
- **Architecture:** RNN, LSTM, Bidirectional LSTM  
- **Activation:** Sigmoid, ReLU, Tanh  
- **Optimizer:** Adam, SGD, RMSProp  
- **Sequence Length:** 25, 50, 100  
- **Gradient Clipping:** Enabled / Disabled  

Metrics used:
- Accuracy  
- F1-score (macro)  
- Training Time (seconds per epoch)

---

## Setup Instructions

### Python Version
Tested on Python 3.10 (Kaggle/CPU environment).

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Code

### Preprocess the Data
```bash
python src/preprocess.py   --input_path "/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"   --output_dir "data/"   --max_words 10000
```

### Train a Model
```bash
python src/train.py   --arch bilstm   --seq_len 50   --activation relu   --optimizer adam   --grad_clip 1.0   --batch_size 32   --epochs 5
```

### Evaluate and Plot
All training logs and metrics are saved to `results/metrics.csv`.  
Visualizations are stored in `results/plots/`.

---

## Results Summary

| Model | Activation | Optimizer | Seq Length | Grad Clipping | Accuracy | F1-score |
|--------|-------------|------------|-------------|----------------|-----------|-----------|
| RNN | ReLU | Adam | 25 | No | 0.82 | 0.80 |
| RNN | ReLU | Adam | 50 | Yes | 0.87 | 0.85 |
| LSTM | Tanh | RMSProp | 100 | Yes | 0.89 | 0.88 |
| BiLSTM | ReLU | Adam | 50 | Yes | 0.91 | 0.90 |
| BiLSTM | Tanh | SGD | 100 | No | 0.86 | 0.84 |

---

## Reproducibility

All runs use fixed random seeds:
```python
import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```
Environment: Kaggle CPU, 16GB RAM.

---

## Report

A complete Overleaf report is available at `Comparative_Analysis_of_RNN_Architectures_for_Sentiment_Classification_Report.pdf` (compiled to PDF).  
It includes Dataset Summary, Model Configuration, Comparative Analysis, Discussion, and Conclusion.

---

## License

This project is for academic use (University of Maryland - MSML Program, Spring 2025).

---
© 2025 Anirudh Krishna
