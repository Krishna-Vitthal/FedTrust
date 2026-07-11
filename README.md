<div align="center">

# ⚡ FedTrust
### Trust-Aware Federated Learning for Defending Against Model Poisoning Attacks

<p align="center">
<b>Dynamic Trust Scoring • Adaptive Client Filtering • Peer-Aware Similarity • Secure Aggregation</b>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Federated Learning](https://img.shields.io/badge/Federated-Learning-success?style=for-the-badge)
![Security](https://img.shields.io/badge/AI-Security-orange?style=for-the-badge)
![Research](https://img.shields.io/badge/IEEE-Research-purple?style=for-the-badge)

</p>

---

> **FedTrust** is a trust-aware Federated Learning framework that dynamically identifies malicious clients, suppresses poisoned updates, and preserves learning performance under model poisoning attacks.

*"Trust is earned. Aggregation isn't."*

</div>

---

# 🚀 Motivation

Traditional **FedAvg** assumes every participating client is honest.

Reality says otherwise.

A handful of malicious clients can completely derail global model convergence using attacks like:

- 🔥 Sign Flipping
- ⚡ Model Scaling
- 🎲 Random Weights
- 🌫 Gaussian Noise
- 💀 Zero Update

FedTrust continuously evaluates every client, estimates its trustworthiness, and lets trusted clients influence the global model more than suspicious ones.

---

# ✨ Key Features

## 🛡 Dynamic Trust Engine

Instead of blindly averaging every update...

FedTrust computes a **Behavior Score** every communication round.

\[
B_i =
0.35S_i+
0.25C_i+
0.20A_i+
0.20N_i
\]

Where

| Score | Meaning |
|--------|----------|
| **S** | Peer-aware gradient similarity |
| **C** | Historical consistency |
| **A** | Global gradient alignment |
| **N** | Robust gradient norm score |

---

## 🧠 Peer-Aware Similarity

Unlike conventional cosine similarity...

FedTrust first groups clients with **similar data distributions** using label histograms.

Similarity is then measured **inside each peer group**, preventing honest Non-IID clients from being unfairly penalized.

---

## 🎯 Adaptive Trust Filtering

Clients with suspicious trust scores are automatically excluded before aggregation.

Instead of using a fixed threshold,

FedTrust computes

```
Threshold = Mean Trust − k × Std. Dev.
```

making filtering adaptive to every training round.

---

## ⚖ Trust-Weighted Aggregation

Instead of

```
FedAvg
```

FedTrust performs

```
Weighted Average
=
Trust × Dataset Size
```

so trustworthy clients dominate the aggregation.

---

## 📈 Trust Recovery

Clients aren't permanently punished.

If a client behaves honestly for several consecutive rounds, its trust gradually recovers.

---

# ⚔ Supported Attacks

| Attack | Supported |
|---------|-----------|
| Sign Flipping | ✅ |
| Model Scaling | ✅ |
| Gaussian Noise | ✅ |
| Random Weights | ✅ |
| Zero Weights | ✅ |
| Mixed Attack Mode | ✅ |

---

# 📊 Dataset Support

- ✅ MNIST
- ✅ IID Split
- ✅ Sorted Non-IID
- ✅ Dirichlet Non-IID

---

# 🏗 Project Structure

```text
FedTrust
│
├── aggregation.py
├── attacks.py
├── client.py
├── config.py
├── dataset.py
├── main.py
├── model.py
├── server.py
├── trust.py
├── utils.py
│
├── results/
│
└── README.md
```

---

# ⚙ Installation

```bash
git clone https://github.com/yourusername/FedTrust.git

cd FedTrust

pip install -r requirements.txt
```

---

# ▶ Run

```bash
python main.py
```

---

# ⚙ Configuration

Everything can be configured inside

```python
config.py
```

Example

```python
NUM_CLIENTS = 10

ROUNDS = 20

LOCAL_EPOCHS = 1

DATA_SPLIT = "dirichlet"

ATTACK_TYPE = "mixed"

USE_TRUST = True
```

---

# 📈 Benchmark Modes

FedTrust supports automated benchmarking.

```python
BENCHMARK_MODE = "comparison"
```

Available modes

- comparison
- ablation
- sensitivity
- heterogeneity
- stress
- runtime
- diagnostics
- full

---

# 📊 Outputs

Automatically generates

- Accuracy Curves
- Loss Curves
- Trust Trajectories
- CSV Results
- Publication Tables
- LaTeX Tables
- Experiment Summaries
- Best Model Checkpoints

---

# 🧠 Trust Pipeline

```text
Client Updates
      │
      ▼
Peer Group Detection
      │
      ▼
Similarity Score
      │
      ▼
Consistency Score
      │
      ▼
Alignment Score
      │
      ▼
Norm Score
      │
      ▼
Behavior Score
      │
      ▼
EMA Trust Update
      │
      ▼
Adaptive Filtering
      │
      ▼
Trust Weighted Aggregation
      │
      ▼
Global Model
```

---

# 📸 Example Results

| Attack | FedAvg | FedTrust |
|----------|---------|-----------|
| Sign Flipping | ❌ | ✅ |
| Scaling | ❌ | ✅ |
| Gaussian | ⚠ | ✅ |
| Random | ❌ | ✅ |
| Mixed | ❌ | ✅ |

---

# 🔬 Research Contributions

- Dynamic trust estimation for Federated Learning
- Peer-aware similarity for Non-IID environments
- Adaptive trust filtering
- Trust-weighted secure aggregation
- Gradient norm anomaly detection
- Trust recovery mechanism
- Comprehensive benchmarking framework

---

# 📚 Citation

```bibtex
@article{fedtrust2026,
  title={FedTrust: A Dynamic Trust-Based Federated Learning Framework Against Model Poisoning Attacks},
  author={Krishna Vitthal},
  year={2026}
}
```

---

# 🤝 Contributing

Contributions are welcome!

Feel free to open an issue or submit a pull request.

---

<div align="center">

### ⭐ If this project helped you, consider giving it a star.

**Secure Learning. Smarter Aggregation. Trusted Intelligence.**

</div>
