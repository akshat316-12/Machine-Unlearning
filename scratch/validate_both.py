import sys
sys.path.insert(0, ".")
import app
client = app.app.test_client()

# Test First-Order
client.post('/reset')
res1 = client.post('/unlearn/first-order')
m1 = res1.get_json()['metrics']
print("First-Order:", m1)

# Test Second-Order
client.post('/reset')
res2 = client.post('/unlearn/second-order', json={"damping": 0.05}, content_type='application/json')
m2 = res2.get_json()['metrics']
print("Second-Order:", m2)

# Validate both
for name, m in [("First-Order", m1), ("Second-Order", m2)]:
    r_ok = m["retain_accuracy"] > 85
    f_ok = m["forget_accuracy"] <= 25
    mia_ok = abs(m["mia_roc_auc"] - 0.50) <= 0.08
    status = "PASS" if (r_ok and f_ok and mia_ok) else "FAIL"
    print(f"{name}: R={m['retain_accuracy']}% [{'ok' if r_ok else 'FAIL'}] F={m['forget_accuracy']}% [{'ok' if f_ok else 'FAIL'}] MIA={m['mia_roc_auc']} [{'ok' if mia_ok else 'FAIL'}] => {status}")
