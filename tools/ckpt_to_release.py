#!/usr/bin/env python3
"""Mini AI: старые попытки моделей → GitHub Release (личный репо), потом удалить с сервера.
Оставляет на сервере: модель из book_school.json (текущая) и лучшую попытку текущей и прошлой четверти."""
import json, os, re, sys, hashlib, urllib.request as u
from pathlib import Path
ROOT = Path('/home/hermes/Developer/mini-ai'); M = ROOT / 'server_data/models'
tok = next(re.search(r'https://[^:]+:([^@]+)@github.com', l).group(1) for l in open('/home/hermes/.git-credentials') if 'github.com' in l)
H = {'Authorization': 'token ' + tok, 'Accept': 'application/vnd.github+json'}
REPO = 'mamazanovfarhod016/mini-ai'; TAG = 'checkpoints-20261020'
st = json.load(open(ROOT / 'server_data/book_school.json'))
keep = {Path(st['model']).name}
g, q = st['grade'], st['quarter']
for (gg, qq) in [(g, q), (g, q - 1) if q > 1 else (g - 1, 4)]:
    same = [h for h in st['history'] if h['grade'] == gg and h['quarter'] == qq]
    if same:
        keep.add(Path(max(same, key=lambda h: h['acc_cur'] + h.get('acc_past', 0))['model']).name)
cands = sorted(p for p in M.glob('*.json') if p.name not in keep)
print('keep:', sorted(keep)); print('move:', [p.name for p in cands])
if '--go' not in sys.argv: sys.exit(0)
def api(m, url, data=None, hdr=None):
    r = u.Request(url, data=data, headers={**H, **(hdr or {})}, method=m)
    return json.load(u.urlopen(r, timeout=600))
try:
    rel = api('GET', f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}')
except Exception:
    rel = api('POST', f'https://api.github.com/repos/{REPO}/releases', json.dumps({'tag_name': TAG, 'target_commitish': 'feat/teach', 'name': 'Чекпоинты обучения по учебникам (до 20.10.2026)', 'body': 'Старые попытки четвертей 1–2 класса и кандидаты. Логи попыток — рядом. Убраны с сервера, чтобы не держать объём.', 'prerelease': True}).encode())
have = {a['name']: a['size'] for a in rel['assets']}
for p in cands:
    for f in [p, p.with_name(p.stem + '_log.txt')]:
        if not f.exists(): continue
        if have.get(f.name) == f.stat().st_size: continue
        up = rel['upload_url'].split('{')[0] + '?name=' + f.name
        a = api('POST', up, f.read_bytes(), {'Content-Type': 'application/octet-stream'})
        print('uploaded', f.name, a['size'])
rel = api('GET', f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}')
have = {a['name']: a['size'] for a in rel['assets']}
freed = 0
for p in cands:
    if have.get(p.name) == p.stat().st_size:
        freed += p.stat().st_size; p.unlink()
    else:
        print('NOT VERIFIED, kept', p.name)
print('release:', rel['html_url'], 'assets', len(have), 'freed MB', round(freed / 1e6))
