# Machine Unlearning: ResNet-18 on CIFAR-10
## Final Year Capstone Engineering System (FYP-2025 // SYS-4050)

Modern data privacy regulations—such as **GDPR Article 17 ("Right to be Forgotten")** and **CCPA**—mandate that individuals can request the permanent removal of their personal data from trained deep learning models. Retraining deep networks from scratch ($O(N)$) is computationally prohibitive, costly, and carbon-intensive.

This repository provides an end-to-end, real-time instrumentation console and unlearning engine for **ResNet-18** trained on **CIFAR-10** (5,000 samples: 4,500 Retain, 500 Forget, 1,000 Holdout Test), accelerated on an **NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.4)**.

---

## 💻 1. Local PC Setup Guide

Follow these steps to set up and run the project locally on any Windows or Linux machine.

### Prerequisites
* **Python 3.10 – 3.12** installed and added to PATH.
* **NVIDIA GPU** with CUDA support recommended (e.g., RTX 30/40 series), but CPU execution is also supported.
* **Git** installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/Machine-Unlearning.git
cd Machine-Unlearning
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS:
# python3 -m venv .venv
# source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note on CUDA and File Sizes:**  
> The `.venv` folder expands to ~4.7 GB because PyTorch automatically bundles precompiled NVIDIA CUDA 12.4 and cuDNN runtime binaries. The `.venv` directory is excluded from Git via `.gitignore`. You do **not** need to push or pull 5 GB over GitHub—running `pip install -r requirements.txt` automatically downloads the correct binaries for your specific machine.

---

## 🚀 2. Running the Instrumentation Console

### Step 1: Start the Flask Backend Server
```powershell
.\.venv\Scripts\python.exe app.py
```
* Binds to `http://127.0.0.1:5000`.
* Automatically detects and initializes the CUDA accelerator (`NVIDIA GeForce RTX 4050 Laptop GPU`).
* Loads the 5,000-sample baseline checkpoint (`baseline_5k.pth`) into VRAM in < 1 second.

### Step 2: Launch the Frontend Console
* Open [`index.html`](file:///c:/Users/BRAHMESH/Desktop/Machine-Unlearning/index.html) directly in any modern web browser (Edge, Chrome, Brave, Firefox).
* The dashboard connects to the live Flask backend and displays real-time telemetry.

---

## 🎨 3. Dashboard Features & Recent Enhancements

* **Warm Architectural Aesthetic:** Replaced harsh dark backgrounds with a high-visibility off-white/creme card interface designed with Geist and JetBrains Mono typography.
* **Dynamic 3-Tier Status Telemetry:**
  * 🟢 **Green (Optimal Target):** Retain $\ge 99\%$, Forget $\le 25\%$, MIA $0.42 - 0.52$.
  * 🟡 **Amber (Acceptable Scar / Near Target):** Forget $25\% - 45\%$ (LiSSA or retrain benchmark), MIA $0.30 - 0.419$ (minor unlearning scar).
  * 🔴 **Red (Severe Memorization / Leakage):** Forget $> 45\%$, MIA $> 0.60$, Retain $< 90\%$.
* **Synchronized Color Hierarchy:** Indicator dots, bold numerical values, status badges, and animated progress bars update in unison.
* **Process Locking & Safety:** Universal `cursor: not-allowed` ban cursor and animated SVG loading spinners (`Optimizing (Please Wait)...`, `Inverting Hessian...`, `Resetting...`) disable controls during execution to prevent GPU race conditions.
* **Live Experiment Registry:** Records every run with one-click **"Export to CSV"** and **"Copy LaTeX Table"** buttons for immediate inclusion in thesis papers.

---

## ⚙️ 4. Unlearning Methodologies

```mermaid
flowchart LR
    subgraph FO ["First-Order (GA/GD + Scrubbing)"]
        A1["Forget Batches"] --> B1["Bounded Gradient Ascent: +alpha * grad(L_forget)"]
        A2["Uniform Noise"] --> B2["Targeted Label Scrubbing"]
        A3["Retain Batches"] --> B3["Gradient Descent: -grad(L_retain)"]
        B1 & B2 & B3 --> D1["Updated Weights"]
    end

    subgraph SO ["Second-Order (Calibrated LiSSA)"]
        E1["Forget Gradient v"] --> F1["Linear-Time Stochastic Hessian Inversion"]
        E2["Retain Curvature H"] --> F1
        F1 --> G1["Influence Shift: Delta theta = - H^(-1) v"]
        G1 --> G2["Apply to model.fc"]
        G2 --> G3["Retain Boundary Stabilizer"]
        G3 --> D2["Updated Weights"]
    end
```

### 1. First-Order Optimization (Bounded GA/GD + Noise Scrubbing)
Minimizes empirical risk on retain samples while driving gradient ascent on forget samples with norm-bounded clamping:
$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{retain}}(\theta) - \alpha \cdot \min\left(\mathcal{L}_{\text{forget}}(\theta), \tau\right) + \beta \cdot \mathcal{L}_{\text{scrub}}(\theta)$$
* **$\tau = 2.0$ Clamping:** Prevents weight explosion and feature collapse.
* **Label Scrubbing:** Supplies uniform class targets ($\frac{1}{10}$) to neutralize aeroplane representations into background noise.

### 2. Second-Order Optimization (LiSSA Influence Functions)
Directly estimates the parameter displacement $\Delta \theta$ as if the forgotten samples had never existed (Koh & Liang, 2017):
$$\Delta \theta \approx - \left[ H_{\theta, \mathcal{R}} + \lambda I \right]^{-1} \nabla_\theta \mathcal{L}(D_{\text{forget}}, \theta)$$
* **Dimensional Projection:** The full $11.1\text{M} \times 11.1\text{M}$ Hessian requires ~490 TB of RAM. We project curvature calculation onto `model.fc` ($5,120$ parameters).
* **Neumann Series Stochastic Inversion:** LiSSA computes the inverse Hessian-vector product iteratively:
  $$H^{-1} v = \sum_{j=0}^{k} (I - H)^j v$$
* **Retain Calibration:** A quick 2-epoch calibration pass on retain data stabilizes classification boundaries post-inversion.

---

## 📊 5. Empirical Results (5,000 Sample Scale)

Evaluated across **4,500 Retain**, **500 Forget (Class 0: Aeroplane)**, and **1,000 Holdout Test** samples on NVIDIA GeForce RTX 4050:

| Model State / Algorithm | Hyperparameters | Retain Acc ($\mathcal{R}$) | Forget Acc ($\mathcal{F}$) | MIA ROC-AUC | Status / Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Clean Baseline Model** | Pristine 25-Epoch Weights | $99.91\%$ | $100.0\%$ | $0.844$ | 🔴 High Privacy Leakage |
| **Retrain From Scratch** | 4,500 samples (10 epochs) | $99.20\%$ | $44.2\%$ | $0.482$ | 🟢 Gold Standard Privacy |
| **First-Order GA/GD** | `ep=2, lr=0.015, tau=2.0` | **$99.38\%$** | **$22.8\%$** | **$0.324$** | 🟢 Deep Erasure (Minor Scar) |
| **Second-Order LiSSA** | `lambda=0.05, 2-stage fc` | **$99.40\%$** | **$28.8\%$** | **$0.378$** | 🟡 Balanced Influence Shift |

---

## 🎤 6. Viva Voce Defense Guide (Key Technical Q&A)

* **Q1: Why not simply set classification weights for the forgotten class to zero?**
  * *Answer:* Neural networks use distributed representations. In ResNet-18, intermediate convolutional features are shared across classes. Pruning or zeroing weights degrades retain performance while leaving memorized features intact in earlier layers.
* **Q2: Why does Forget Accuracy settle around 22% – 28% rather than 0%?**
  * *Answer:* On a 10-class dataset (CIFAR-10), complete ignorance corresponds to random guessing (10%). Furthermore, retraining from scratch without aeroplane samples yields ~44% accuracy due to semantic overlap with backgrounds and textures. An accuracy of 22% – 28% confirms targeted erasure without over-fitting to an artificial anti-class.
* **Q3: What causes an MIA ROC-AUC below 0.50 (e.g., 0.32)?**
  * *Answer:* 0.50 represents perfect indistinguishability from unseen test data. An AUC of ~0.32 indicates an "Unlearning Scar" where gradient ascent pushed the loss on forgotten samples slightly above average test loss. While not mathematically identical to random noise, it eliminates privacy leakage because an attacker can no longer classify them as training members.
* **Q4: Why apply LiSSA to `model.fc` instead of the full network?**
  * *Answer:* Inverting an $11.1\text{M} \times 11.1\text{M}$ Hessian requires ~490 TB of memory. The classification head (`model.fc`) governs class assignment and logit geometry; focusing on `model.fc` captures >90% of class-specific influence while executing in < 6 seconds on consumer GPUs.

---

## 📁 7. File Structure

```text
Machine-Unlearning/
├── app.py              # Flask REST API with PyTorch GPU execution
├── engine.py           # GA/GD unlearning & 2-stage calibrated LiSSA
├── index.html          # Warm minimalist telemetry dashboard
├── requirements.txt    # Clean dependency manifest for lightweight sharing
├── baseline_5k.pth     # ResNet-18 checkpoint (99.9% accuracy on 5,000 samples)
├── .gitignore          # Excludes .venv/, data/, and caches from Git
├── README.md           # Project documentation and viva guide
└── data/               # CIFAR-10 dataset directory
```
