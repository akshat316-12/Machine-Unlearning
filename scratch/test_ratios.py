import sys
sys.path.append('scratch')
from test_perfect_balance import run_balanced_unlearn

for ratio in [0.4, 0.5, 0.6, 0.7]:
    for a in [0.85, 1.0, 1.2]:
        m = run_balanced_unlearn(epochs=4, lr=0.018, alpha=a, scrub_ratio=ratio)
        print(f"ratio={ratio}, a={a} -> Retain={m['retain_accuracy']}%, Forget={m['forget_accuracy']}%, MIA={m['mia_roc_auc']}")
