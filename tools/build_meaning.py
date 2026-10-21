"""Строит базовый смысл слов из Simple Wikipedia (text.jsonl) один раз → meaning_base.npz. Без градиента."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_ai.meaning import Meaning, BASE_NPZ
SRC = os.path.expanduser("~/Developer/mini-ai-runs/memory-core/simplewiki_full/text.jsonl"); LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
M = Meaning(None); t = time.time(); n = 0
with open(SRC) as f:
    for line in f:
        d = json.loads(line); M.read(d[0] if isinstance(d, list) else str(d)); n += 1
        if n >= LIMIT: break
keep = [w for w in M.sem if M.n[w] >= 5]; M.sem = {w: M.sem[w] for w in keep}
M.path = BASE_NPZ; M._dirty = 1; M.save(force=True)
print(f"{n} sentences, {len(keep)} words, {time.time()-t:.0f}s -> {BASE_NPZ}, {os.path.getsize(BASE_NPZ)/1e6:.0f} MB")
