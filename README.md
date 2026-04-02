# Deep Learning Applications (DLA) — Lab Projects

This repository contains the laboratory assignments for the **Deep Learning Applications** course. Each lab focuses on a different area of deep learning and reinforcement learning, building progressively from RL fundamentals to transfer learning and adversarial robustness.

---

## Repository Structure

```
DLA/
├── Lab2/              # Deep Reinforcement Learning (REINFORCE, DQL)
├── Lab3/              # Transfer Learning & Prompt Learning
├── Lab4/              # Adversarial Learning and OOD Detection
├── requirements.txt   # Python dependencies
└── logs/              # Training logs and results
```

---

## Setup Instructions

### Prerequisites
- Python **3.10**
- CUDA 12.8 (for GPU acceleration)
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/BrunoMartelli01/DLA.git
cd DLA
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

> **Note:** This project uses PyTorch with CUDA 12.8. If you have a different CUDA version, modify the torch installation line in `requirements.txt` accordingly.

---

## Running the Labs

### Lab 2 — Deep Reinforcement Learning (REINFORCE & DQL)

Location: `Lab2/`

This lab implements reinforcement learning algorithms, progressing from basic policy gradient methods to vision-based deep Q-learning:
- **REINFORCE** — Policy gradient with and without a value baseline on `CartPole-v1`
- **Deep Q-Learning (DQL)** — CNN-based agent for the `CarRacing-v3` environment

**Run the notebook:**
```bash
cd Lab2
jupyter notebook Lab2.ipynb
```

**Run standalone scripts:**
```bash
python Lab2/DQL.py        # Train DQL agent
```

See [Lab2/README.md](Lab2/README.md) for detailed information.

---

### Lab 3 — Transfer Learning & Prompt Learning

Location: `Lab3/`

This lab explores parameter-efficient adaptation of pre-trained models, progressing from feature extraction to prompt learning:
- **Exercise 1:** Sentiment analysis on Rotten Tomatoes using DistilBERT feature extraction + SVM classifier
- **Exercise 2:** Full fine-tuning of DistilBERT for sequence classification with HuggingFace `Trainer`
- **Exercise 3:** Parameter-efficient adaptation of CLIP to `Flowers102` via **CoOp** (Context Optimization), training only 16 learnable prompt tokens while keeping all CLIP weights frozen

**Run the notebook:**
```bash
cd Lab3
jupyter notebook Lab3.ipynb
```

See [Lab3/README.md](Lab3/README.md) for detailed information.

---

### Lab 4 — Adversarial Learning and OOD Detection

Location: `Lab4/`

This lab develops a methodology for detecting out-of-distribution (OOD) samples and measuring detection quality. It also experiments with incorporating adversarial examples during training to improve model robustness.

*   **Exercise 1:** Builds a simple OOD detection pipeline using CIFAR-10 (ID) and FakeData (OOD), trains a CNN classifier and a convolutional autoencoder, and evaluates them with AUROC, AUPR, FPR@95TPR, and Detection Error@95TPR.
*   **Exercise 2:** Implements FGSM adversarial attacks and adversarial training, measuring the trade-off between clean accuracy, adversarial robustness, and OOD detection quality.
*   **Exercise 3:** Implements ODIN (temperature scaling + input preprocessing) as a stronger OOD detector and compares it against the baseline pipeline.

**Run the notebook:**
```bash
cd Lab4
jupyter notebook Lab4.ipynb
```

**Pre-trained models:**
The directory includes two model checkpoints:
- `cifar10_clean.pth` CNN trained on clean CIFAR-10 data
- `cifar10_adversarial.pth` CNN trained with FGSM adversarial augmentation

See [Lab4/README.md](Lab4/README.md) for detailed information.

---


## License

This repository is for educational purposes as part of the Deep Learning Applications course.

## Author

Bruno Martelli ([BrunoMartelli01](https://github.com/BrunoMartelli01))
