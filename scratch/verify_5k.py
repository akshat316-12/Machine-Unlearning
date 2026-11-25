import sys
sys.path.insert(0, ".")
import app

client = app.app.test_client()

print("--- 1. Testing Initial Baseline Evaluation (5,000 samples) ---")
res = client.get('/evaluate')
initial = res.get_json()['current_metrics']
print("Initial baseline metrics:", initial)

print("\n--- 2. Testing First-Order Unlearning (2 epochs, lr=0.015) ---")
client.post('/reset')
res_fo = client.post('/unlearn/first-order', json={'epochs': 2, 'lr': 0.015})
m_fo = res_fo.get_json()['metrics']
print("First-Order metrics:", m_fo)

print("\n--- 3. Testing Second-Order LiSSA Unlearning (damping=0.05) ---")
client.post('/reset')
res_so = client.post('/unlearn/second-order', json={'damping': 0.05})
m_so = res_so.get_json()['metrics']
print("Second-Order metrics:", m_so)

print("\n--- 4. Benchmarks Check ---")
for name, m in [("First-Order", m_fo), ("Second-Order", m_so)]:
    r_ok = m['retain_accuracy'] >= 85.0
    f_ok = m['forget_accuracy'] <= 28.0
    mia_ok = abs(m['mia_roc_auc'] - 0.50) <= 0.08
    print(f"{name}: Retain={m['retain_accuracy']}% [{'PASS' if r_ok else 'FAIL'}] | Forget={m['forget_accuracy']}% [{'PASS' if f_ok else 'FAIL'}] | MIA={m['mia_roc_auc']} [{'PASS' if mia_ok else 'FAIL'}]")
