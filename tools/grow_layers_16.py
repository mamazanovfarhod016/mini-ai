"""Добавить слои 14 → 16 к модели роста (новые слои = пропуск x→x, ответы не меняются; экзамен до/после)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, '/home/hermes/Developer/mini-ai')
from mini_ai.transformer import export_model, grow_params, init_params, load_model, n_params
src = '/home/hermes/Developer/mini-ai/server_data/models/_grow31m.json'
dst = '/home/hermes/Developer/mini-ai/server_data/models/_grow31m_16l.json'
m = json.loads(Path(src).read_text(encoding='utf-8'))
p_old, cfg_old, tok, _ = load_model(m)
skip = {"format", "config", "vocab", "max_q", "tokenizer", "order", "shapes", "weights", "merges"}
meta = {k: v for k, v in m.items() if k not in skip}
max_q = m.get("max_q", 24)
del m
p_new, cfg_new = init_params(len(tok), d=cfg_old['d'], n_layer=16, n_head=cfg_old['n_head'], block=cfg_old['block'],
                             pos=cfg_old.get('pos', 'learned'))
p_new = grow_params(p_new, cfg_new, p_old, cfg_old)
N = n_params(p_new)
meta.update({"n_params": N, "grown_at": time.strftime("%Y-%m-%d %H:%M"), "grown_layers": f"{cfg_old['n_layer']}->16"})
Path(dst).write_text(json.dumps(export_model(p_new, cfg_new, tok, meta, tokenizer_type="bpe", max_q=max_q, merges=tok.merges),
                                ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(cfg_new, f"{N:,}", round(Path(dst).stat().st_size / 1e6), "MB")
from server.book_school import exam, stages
items = [x for p in stages() if (p["grade"], p["quarter"]) <= (2, 2) for x in p["test"]][:200]
print("exam 14 layers", exam(src, items)[0], "16 layers", exam(dst, items)[0])
