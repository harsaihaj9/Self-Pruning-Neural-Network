# Self-Pruning Neural Network for CIFAR-10  
### Tredence AI Engineering Internship – Case Study Submission

## Overview
This project implements a **self-pruning neural network** in PyTorch that learns to remove its own weak connections during training.

Instead of pruning weights after training, each weight is paired with a learnable **gate parameter**. These gates decide whether a connection should stay active or be suppressed.

The model is trained on the CIFAR-10 image classification dataset and evaluated for:
- Test Accuracy
- Sparsity Level
- Accuracy vs Sparsity Trade-off

---

## Problem Statement
Large neural networks are powerful but expensive to deploy due to:
- High memory usage
- Slow inference speed
- Large compute requirements

This project solves that by allowing the network to **shrink itself automatically** while learning.

---

## Core Idea
Each weight has a learnable gate:

`gate = sigmoid(gate_score)`

The effective weight becomes:

`W' = W * gate`

If gate approaches **0**, that connection is effectively removed.

---

## Loss Function
The total loss combines:
1. Classification Loss (Cross Entropy)
2. Sparsity Loss (L1 penalty on gates)

`Loss = CE + λ * Σ(gates)`

Where:
- Low λ → Better accuracy, less pruning
- High λ → More pruning, possible accuracy drop

---

## Project Structure
```bash
.
├── main.py
├── requirements.txt
├── README.md
├── report.md
```

## Tech Stack
- Python
- PyTorch
- Torchvision
- Matplotlib
- Pandas

## Dataset
CIFAR-10 contains 60,000 color images across 10 classes and is downloaded automatically.

## How to Run
```bash
pip install -r requirements.txt
python main.py
```

## Output Generated
After execution:
- `results.csv`
- `gate_distribution.png`

## Why This Works
L1 regularization encourages many gate values to become zero or near zero.

Benefits:
- Fewer active parameters
- Smaller model
- Faster inference
- Lower memory usage

## Future Improvements
- Use CNN instead of MLP
- Structured neuron pruning
- Fine-tuning after pruning
- ONNX deployment

## Author
**Harsaihaj Singh**  
Computer Engineering Student  
Thapar Institute of Engineering and Technology

GitHub: https://github.com/harsaihaj9
