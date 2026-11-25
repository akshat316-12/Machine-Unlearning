import sys
sys.path.insert(0, ".")
import app

client = app.app.test_client()

print("1. Status check:")
res = client.get('/status')
print(res.get_json())

print("\n2. Initial evaluate:")
res = client.get('/evaluate')
print(res.get_json()['current_metrics'])

print("\n3. Testing Fisher-EWC + Distillation unlearning...")
client.post('/reset')
res = client.post('/unlearn/fisher-ewc', json={'epochs': 2, 'lr': 0.015, 'ewc_lambda': 0.3, 'kd_weight': 0.3})
m = res.get_json()['metrics']
print("Fisher EWC Metrics:", m)

print("\n4. Testing First-Order unlearning...")
client.post('/reset')
res = client.post('/unlearn/first-order', json={'epochs': 2, 'lr': 0.015})
m = res.get_json()['metrics']
print("First-Order Metrics:", m)

print("\n5. Testing Second-Order LiSSA unlearning...")
client.post('/reset')
res = client.post('/unlearn/second-order', json={'damping': 0.05})
m = res.get_json()['metrics']
print("Second-Order Metrics:", m)

print("\nAll 3 unlearning endpoints verified!")
