import requests, json, sys
from pathlib import Path

p = Path('Inputs/prometheus_eval_01.json')
if not p.exists():
    print("Input file not found:", p)
    sys.exit(2)

try:
    data = json.loads(p.read_text(encoding='utf-8'))
except Exception as e:
    print('Failed to read JSON:', e)
    sys.exit(2)

url = 'http://127.0.0.1:8000/evaluate'
print('POSTing to', url)
try:
    r = requests.post(url, json=data, timeout=60)
    print('STATUS', r.status_code)
    print(r.text)
except Exception as e:
    print('ERROR', e)
    sys.exit(1)
