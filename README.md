# Machine Unlearning
## 🎯 1. Project Goal & Target Benchmarks
Modern data privacy laws (such as GDPR's *"Right to be Forgotten"*) require AI models to delete specific user records or subsets upon request without retraining the entire neural network from scratch.

This prototype implements selective data erasure on a **ResNet-18** classifier trained on **CIFAR-10** subsets (Retain Set $\mathcal{R}$: indices `0–400`, Forget Set $\mathcal{F}$: indices `400–500`). For a successful unlearning run, the model must achieve three target metrics simultaneously:

1. **Retain Accuracy > 85%:** Maintain high classification accuracy on the remaining training data (no catastrophic forgetting).
2. **Forget Accuracy ~ 10%:** Drop classification accuracy on the erased subset down to random-guessing levels (10% across 10 classes).
3. **MIA ROC-AUC Score ~ 0.50:** Reduce the Membership Inference Attack (MIA) shadow classifier's ability to distinguish whether forgotten data was ever in the training set.

---

## 🚀 2. How to Setup & Run the Prototype

### Step 1: Clone the Repository
```bash
git clone https://github.com/akshat316-12/Machine-Unlearning.git
cd Machine-Unlearning
```

### Step 2: Install Required Dependencies
Ensure **Python 3.10+** is installed (NVIDIA CUDA GPU recommended, falls back to CPU automatically), then run:
```bash
pip install torch torchvision flask flask-cors scikit-learn numpy
```

### Step 3: Start the Flask Backend Server
Launch the API server from the project root directory:
```bash
python app.py
```
* **Startup Behavior:** The server will automatically download CIFAR-10 into `./data` (if missing) and execute a baseline ResNet-18 training loop so the model memorizes the subsets (~95% accuracy).
* **Ready State:** Keep this terminal open and wait until you see:
  ```text
  * Running on http://127.0.0.1:5000
  ```

### Step 4: Open the Web Dashboard
* **Option A (VS Code Live Server — Recommended):** Right-click `index.html` in VS Code and select **"Open with Live Server"** (opens at `http://127.0.0.1:5500/index.html`).
* **Option B (Direct Browser Open):** Double-click `index.html` in your file explorer to open it directly in any browser.

### Step 5: Test & Operate the Dashboard
1. Verify the top-right badge displays **`Live Flask`**.
2. Adjust **Epochs** and **Learning Rate** (for First-Order Gradient Ascent/Descent) or **Hessian Damping Factor** (for Second-Order LiSSA) and click the respective **Run** button.
3. Track real-time updates in the **Execution Logs** terminal and the three metric cards at the top.

> **⚠️ Crucial Testing Rule:** Every unlearning click mutates the active model weights in memory. If a test run causes catastrophic forgetting (Retain Accuracy drops to ~10%), press `Ctrl + C` in the terminal and restart `python app.py` to reset back to the clean 95%-accurate baseline model before testing new parameters.

---

## 🛠️ 3. Handoff Status: What Works & What Needs to Be Done Next

### ✅ Already Completed & Working
* **End-to-End Architecture:** Live communication between PyTorch (`engine.py`), Flask REST API (`app.py`), and the interactive frontend (`index.html`).
* **First-Order Unlearning Engine (`engine.py`):** Gradient Ascent on $\mathcal{F}$ paired with Gradient Descent anchoring on $\mathcal{R}$ (with in-place `ReLU` layers disabled to prevent PyTorch autograd crashes).
* **Second-Order Unlearning Engine (`engine.py`):** Damped Hessian Influence Functions using LiSSA approximation targeting the final classification layer (`model.fc`).
* **Live Evaluation Pipeline:** Automated computation of Retain Accuracy, Forget Accuracy, and MIA ROC-AUC scores after each run.

### 🔧 Action Items to Complete Next (To-Do List)

#### 1. Fix Catastrophic Forgetting & Tune Hyperparameters (`engine.py`)
* **The Issue:** Currently, `lr = 0.0001` does almost nothing (Forget Accuracy stays at ~95%), while `lr = 0.01` over-corrects and destroys the whole network (Retain Accuracy crashes to ~11%).
* **What to Do:**
  * Test middle-ground dashboard parameters first: **Epochs = `1` or `2`**, **Learning Rate = `0.0005` to `0.002`**.
  * In `engine.py`, scale down the forget loss using a weighting hyperparameter $\lambda$ (e.g., `0.1`) so the retain set anchors the weights more strongly:
    ```python
    total_loss = retain_loss - 0.1 * forget_loss
    ```
  * Add gradient clipping right before `optimizer.step()` to stop gradient ascent from exploding the weights:
    ```python
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    ```

#### 2. Save Baseline Weights & Add a Reset Endpoint (`app.py` & `index.html`)
* **The Issue:** Restarting `app.py` forces ResNet-18 to retrain from scratch every single time, slowing down testing.
* **What to Do:**
  * Save the trained baseline weights (`torch.save(model.state_dict(), "baseline.pth")`) after the first run and load them automatically on subsequent startups.
  * Add a `/reset` POST endpoint in `app.py` (which reloads `baseline.pth`) and wire a **"Reset Model"** button in `index.html` so hyperparameters can be tested rapidly without restarting the terminal.

#### 3. Scale Up Data Subsets & Optimize Evaluation Speed (`app.py` & `engine.py`)
* **What to Do:**
  * Expand from the initial 500-image prototype slice (`0–400` retain, `400–500` forget) to full CIFAR-10 class-wise unlearning (e.g., forgetting an entire class like "airplanes" or larger multi-batch subsets).
  * Optimize or toggle the Membership Inference Attack (MIA) logistic regression loop in `evaluate_metrics()` so UI responses stay fast during multi-epoch runs.

#### 4. Dashboard UI Cleanup (`index.html`)
* **What to Do:** Replace raw LaTeX text in the card titles (`$\mathcal{R}$`, `$\mathcal{F}$`, `$\text{model.fc}$`) with clean HTML formatting or MathJax so they render properly in the browser.

---

## 📁 4. Repository File Map
```text
├── app.py              # Flask server, API endpoints, CIFAR-10 loaders & baseline training
├── engine.py           # First-order, Second-order (LiSSA) unlearning & MIA evaluation logic
├── index.html          # Interactive web dashboard UI, parameter controls & execution logs
└── data/               # Auto-downloaded CIFAR-10 dataset directory
```
