import sys, os; sys.path.insert(0, '.')
exec(open("tools/exp44_sciq.py").read().split("rng = np.random.default_rng(0); res = {}")[0])
for q, gold, bad in [("The electrode at which oxidation occurs is called?", "the anode", "cathode"),
                     ("Natural gas is the predominately made up of?", "methane", "hydrogen"),
                     ("What contains positive protons and neutral neutrons?", "nucleus", "electrons")]:
    qw = set(content_words(toks(q))); print("\nQ:", q, "| qw =", qw)
    for o in (gold, bad):
        ow = set(content_words(toks(o))); cand = set.union(*[inv.get(w, set()) for w in ow])
        rows = []
        for i in cand:
            sw = set(tok_s[i]); hit = qw & sw
            if ow & sw and hit: rows.append((w_idf(hit) / w_idf(qw), sents[i][:110], sorted(hit)))
        rows.sort(reverse=True)
        print(f"  «{o}»: предложений с вариантом {len(cand)}, лучшие:")
        for r in rows[:2]: print(f"     cov={r[0]:.2f} {r[2]} | {r[1]}")
