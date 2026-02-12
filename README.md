# Final-Year-Project

Final year project for the *University of Westminster* (BSc Data Science and Analytics). This repository contains the code for a *Privacy-Preserving Machine Learning* system that explores the tradeoff between *Differential Privacy (DP)* and *Homomorphic Encryption (HE)* when training and evaluating neural networks on sensitive data.

---

## Overview

The project implements and visualises:

- *Differential Privacy* via [Opacus](https://opacus.ai/) — private training of CNNs with \((\varepsilon, \delta)\)-DP guarantees.
- *Homomorphic Encryption* via [TenSEAL](https://github.com/OpenMined/TenSEAL) — encrypted inference so predictions can be computed on encrypted data.
- An *interactive Streamlit dashboard* to run experiments and compare plaintext vs encrypted inference accuracy across different privacy budgets.

The main use case is *MNIST*: you can train a DP model, then optionally run real encrypted inference on the test set and compare accuracy vs a plaintext baseline.

---

## Features

- *Privacy-Accuracy Tradeoff Dashboard* — Run multiple experiments with different \(\varepsilon\) values and compare:
  - Plaintext accuracy (Opacus DP training, standard inference)
  - Real encrypted inference accuracy (hybrid: plain conv + TenSEAL-encrypted fully connected layers)
- *Two model types*:
  - *CNN* — Standard CNN with tanh activations (plaintext only).
  - *HE-friendly CNN* — Square activations for compatibility with TenSEAL’s CKKS scheme (supports encrypted inference).
- *HE simulation (training)* — Optional CKKS-like noise injection on training data to simulate HE effects during training.
- *Real encrypted inference* — TenSEAL-based encrypted evaluation on a subset or full test set.
- *Educational HE module* (Homomorphic_Encryption.py) — From-scratch BFV-style homomorphic encryption (encrypt, decrypt, add/mul with plaintext) for understanding HE basics.

---

## Project Structure


Final-Year-Project/
├── dashboard.py              # Streamlit app: Privacy-Accuracy Tradeoff Dashboard
├── privacy_ml_framework.py   # Core: Opacus DP training, TenSEAL inference, MNIST, plotting
├── Homomorphic_Encryption.py # Educational BFV-style HE (polynomial ops, keygen, encrypt/decrypt)
├── main.py                   # (Reserved for CLI or other entry points)
├── requirements.txt          # Python dependencies
└── README.md                 # This file


---

## Prerequisites

- *Python* 3.9+ (tested with 3.9–3.13)
- *pip* for installing dependencies

---

## Installation

1. Clone the repository:

   bash
   git clone <repository-url>
   cd Final-Year-Project
   

2. Create and activate a virtual environment (recommended):

   bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   

3. Install dependencies:

   bash
   pip install -r requirements.txt
   

   Main dependencies: streamlit, numpy, pandas, matplotlib, seaborn, scikit-learn, plotly, opacus, torch, torchvision, tenseal.

---

## Usage

1. Start the dashboard:

   bash
   streamlit run dashboard.py
   

2. In the browser:
   - Choose *ML Model*: he_cnn (for encrypted inference) or cnn.
   - Set *Epsilon range, **Training epochs, and **Number of experiments*.
   - Optionally enable *HE Simulation (Training Phase)* to inject CKKS-like noise during training.
   - Optionally enable *Run Real Encrypted Inference (Test Phase)* to evaluate with TenSEAL on the test set (small sample or full set).
   - Click *Run Experiments* and inspect the privacy–accuracy plot and results table.

---

## Technical Notes

- *Opacus*: DP-SGD with a target \(\varepsilon\) (and \(\delta = 10^{-5}\)); gradient clipping and noise calibrated to the chosen \(\varepsilon\).
- *TenSEAL: CKKS scheme; encrypted inference is **hybrid*: convolution + square activation run in plaintext on the client, then encrypted feature vector is sent to the server for encrypted FC layers (FC → square → FC); result is decrypted for the final class.
- *Dataset*: MNIST (via sklearn.datasets.fetch_openml); the dashboard subsamples to 2000 training points for responsiveness.
