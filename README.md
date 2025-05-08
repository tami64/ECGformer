# ECGformer
ECGformer: Leveraging transformer for ECG heartbeat arrhythmia classification


This repository contains code to train and evaluate a Transformer-based deep learning model for heartbeat classification using the MIT-BIH Arrhythmia dataset. The model is implemented in TensorFlow/Keras and leverages multi-head self-attention for time-series signal analysis.

## 🧠 Project Overview

The goal is to classify heartbeats into five categories (`N`, `S`, `V`, `F`, `Q`) using 1D ECG signals. The model architecture is based on Transformer encoders followed by an MLP classification head. It processes standardized ECG input and evaluates performance using classification metrics and visualizations.

This implementation is based on the following reference:

> Akan, T., Alp, S., & Bhuiyan, M. A. N. (2023, December). **ECGformer: Leveraging transformer for ECG heartbeat arrhythmia classification**. In *2023 International Conference on Computational Science and Computational Intelligence (CSCI)* (pp. 1412–1417). IEEE. [https://doi.org/10.1109/CSCI62032.2023.00231](https://doi.org/10.1109/CSCI62032.2023.00231)

If you use this code or adapt it for your own research, please consider citing the original paper.


## 📁 Dataset

- **Source**: [MIT-BIH Arrhythmia Dataset](https://physionet.org/content/mitdb/1.0.0/)
- **Expected Files** (Just 10 samples):  
  - `mitbih_train.csv`  
  - `mitbih_test.csv`

These should be placed in the same directory as the script. Each row represents an ECG sample, where the last column is the label.

## 📦 Dependencies

Make sure to install the following Python libraries:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow

