import sys,time; sys.path.insert(0,'.')
import tools.exp21b_wordlevel as m
import pandas as pd, numpy as np
from mini_ai.transformer import BOS,EOS,SEP,PAD,init_params,loss_and_grads,clip_grads,AdamW,lr_schedule,n_params,KVModel
train_df = pd.read_parquet('data/babi/task_1-single_supporting_fact_train.parquet')
test_df = pd.read_parquet('data/babi/task_1-single_supporting_fact_test.parquet')
def pair(r): return (m.tok_words(r['passage'])+' <q> '+m.tok_words(r['question']), r['answer'].strip().lower())
train_pairs=[pair(r) for _,r in train_df.iterrows()]
test_pairs=[pair(r) for _,r in test_df.iterrows()]
all_words=set()
for q,a in train_pairs+test_pairs: all_words.update(q.split()); all_words.update(a.split())
all_words.add('<q>')
tok = m.WordTokenizer(all_words); V=len(tok)
def encode(q,a):
    qi=tok.encode(q)[:60]; ai=tok.encode(a)[:5]
    ids=[BOS]+qi+[SEP]+ai+[EOS]; mask=[0]*(2+len(qi))+[1]*(len(ai)+1)
    return ids,mask
enc=[encode(q,a) for q,a in train_pairs]
p,cfg = init_params(V, d=128, n_layer=4, n_head=8, pos='rope', seed=0)
print('params', n_params(p), flush=True)
opt = AdamW(p, lr=5e-3, wd=0.01)
rng=np.random.default_rng(0)
BATCH=32
steps_per_epoch = len(enc)//BATCH
total_steps = 80*steps_per_epoch
step=0; t0=time.time()
for ep in range(80):
    order=rng.permutation(len(enc)); epl=0
    for bi in range(0,len(order),BATCH):
        idxs=order[bi:bi+BATCH]; batch=[enc[i] for i in idxs]
        T=max(len(ids) for ids,_ in batch)
        X=np.full((len(batch),T),PAD,dtype=np.int64); M=np.zeros((len(batch),T))
        for i,(ids,mask) in enumerate(batch): X[i,:len(ids)]=ids; M[i,:len(mask)]=mask
        targets=np.zeros_like(X); targets[:,:-1]=X[:,1:]
        mask_t=np.zeros_like(M); mask_t[:,:-1]=M[:,1:]
        lr = lr_schedule(step, total_steps, 5e-3, warmup=max(5,total_steps//20))
        loss,grads = loss_and_grads(p,cfg,X,targets,mask_t)
        clip_grads(grads,1.0); opt.step(p,grads,lr); epl+=loss; step+=1
    if ep%3==0 or ep==79:
        kv=KVModel(p,cfg)
        rng_t=np.random.default_rng(1)
        idxs=rng_t.choice(len(test_pairs),size=150,replace=False)
        ok=0
        for i in idxs:
            q,gold = test_pairs[i]
            ids=[BOS]+tok.encode(q)[:60]+[SEP]
            out,_ = kv.generate(ids, max_new=5, temperature=0.0)
            pred=tok.decode(out).strip()
            ok += pred==gold
        acc = ok/150
        print(f'ep={ep} loss={epl/steps_per_epoch:.3f} acc={acc:.2f} t={time.time()-t0:.0f}s', flush=True)
        if acc>=0.95:
            print('REACHED 95% at epoch', ep, 'time', time.time()-t0, flush=True)
            break
print('DONE total time', time.time()-t0, flush=True)
